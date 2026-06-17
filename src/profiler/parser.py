import re


class MapiProParser:
    """
    Hardware-Aware Static Profiler for MSP430FR6989 (FRAM-based).
    Maps C code to physical energy/cycle costs for MAPI-PRO simulation.
    """

    def __init__(self):
        # Energy Constants (nJ) - MSP430FR6989 Specs
        self.ENERGY_SRAM_READ = 5200.0
        self.ENERGY_SRAM_WRITE = 5400.0
        self.ENERGY_FRAM_READ = 10500.0
        self.ENERGY_FRAM_WRITE = 13500.0
        self.ENERGY_LOGIC_OP = 2500.0

        # Latency (Clock Cycles @ 16MHz)
        self.LATENCY_SRAM = 1
        self.LATENCY_FRAM = 3
        self.LATENCY_LOGIC = 1

        self.predicted_algo = "Unknown"

        # STRIPPED SIGNATURES: Removed generic C vars (temp, index) and BEEBS global functions
        self.signatures = {
            "CRC-16": ["crc", "icrc", "poly", "cword"],
            "QuickSort": ["quicksort", "partition", "pivot", "istack", "jstack", "nstack"],
            "Dijkstra": ["dijkstra", "minnd", "rgnnodes"],
            "BubbleSort": ["bubblesort", "numelems"],
            "Cubic": ["solvecubic", "r2_q3"],
            "InsertSort": ["insertsort"],
            "MergeSort": ["mergesort", "mergesortr", "binarylast", "range_length"],
            "Prime": ["prime", "divides"],
            "Hash": ["sglib", "hashed", "htab", "ilist"],
            "StringSearch": ["stringsearch1", "prep1", "exec1", "prep2", "exec2"],
            "Recursion": ["anka(", "kalle("],
            "StrStr": ["strstr", "phaystack", "pneedle", "rhaystack", "rneedle", "foundneedle"],
            "WikiSort": ["wikisort", "wikimerge", "blockswap", "floorpoweroftwo"],
            "Huffman": ["huffbench", "compdecomp", "heap_adjust"],
            "FibCall": ["fibcall", "fnew", "fold", "832040", "apsim_loop"]
        }

    def profile_line(self, line, mapping_config="HYBRID"):
        """
        Calculates the physical 'cost' of executing one line of C code.
        """
        line = line.strip()

        # Costs 0 if it's just structural syntax
        if not line or line in ["{", "}", "(", ")", ";"]:
            return 0.0, 0

        energy = 0.0
        cycles = 0

        # Identify instruction type
        is_write = "=" in line and "==" not in line and "!=" not in line and "<=" not in line and ">=" not in line

        # Check for data-heavy operations (Added string search, strstr, WikiSort, and Huffman arrays/buffers)
        is_read = any(keyword in line for keyword in
                      ["data[", "arr[", "input", "lin[", "rgnNodes", "array[", "x[", "a[", "array1[", "buffer[",
                       "memcpy", "htab[", "heap[", "malloc_beebs", "buf[", "search[", "strlen", "text", "substr",
                       "phaystack", "pneedle", "rhaystack", "rneedle", "cache[", "wikimerge",
                       "freq[", "link[", "code[", "clen[", "heap2[", "outc[", "comp[", "cptr", "dptr"])

        # Expanded logic operations to capture standard library math functions, modulo (%), and sqrt()
        is_logic = any(op in line for op in
                       ["^", ">>", "<<", "&", "|", "+", "-", "*", "++", "--", "%", "pow(", "acos(", "sqrt(", "cos(",
                        "fabs("])

        # MAPI-PRO Hybrid Memory Mapping Logic
        # Large arrays, hash tables, text strings, compression tables, and custom heaps are mapped to FRAM; scalars to SRAM
        if any(buf in line for buf in
               ["arr[", "lin[", "AdjMatrix", "rgnNodes", "array[", "x[", "a[", "array1[", "buffer[", "memcpy", "htab[",
                "heap[", "buf[", "search[", "text", "substr", "phaystack", "pneedle", "rhaystack", "rneedle", "cache[",
                "freq[", "link[", "code[", "clen[", "heap2[", "outc[", "comp[", "heap_ptr", "heap_end"]):
            mem_read, mem_write, mem_lat = self.ENERGY_FRAM_READ, self.ENERGY_FRAM_WRITE, self.LATENCY_FRAM
        else:
            mem_read, mem_write, mem_lat = self.ENERGY_SRAM_READ, self.ENERGY_SRAM_WRITE, self.LATENCY_SRAM

        if is_write:
            energy += mem_write
            cycles += mem_lat
        if is_read:
            energy += mem_read
            cycles += mem_lat
        if is_logic:
            energy += self.ENERGY_LOGIC_OP
            cycles += self.LATENCY_LOGIC

        # Base instruction overhead (fetch/decode)
        energy += 1000.0
        cycles += 1

        return energy, cycles

    def get_checkpoint_cost(self, state_size_bytes=32):
        """
        The cost of saving state to Non-Volatile FRAM.
        Increasing this value encourages the agent to be more efficient.
        """
        # Saving state requires writing to FRAM
        energy = state_size_bytes * self.ENERGY_FRAM_WRITE
        cycles = state_size_bytes * self.LATENCY_FRAM
        return energy, cycles

    def get_cyclomatic_complexity(self, code_block):
        """
        Static analysis of branching complexity.
        Used for telemetry only; ignored by the ML decision engine for max gain.
        """
        decision_points = ['if', 'else', 'for', 'while', 'switch', 'case', '&&', '||']
        score = 1
        for point in decision_points:
            if point in code_block:
                score += 1
        return score

    def load_c_file(self, file_path):
        """
        Parses C file into clean instructions and identifies the algorithm via frequency scoring.
        """
        clean_instructions = []
        in_multiline_comment = False

        try:
            with open(file_path, 'r') as f:
                content = f.read()

                # IDENTIFY ALGO: Frequency Scoring System
                content_lower = content.lower()
                algo_scores = {algo: 0 for algo in self.signatures.keys()}

                for algo, keywords in self.signatures.items():
                    for key in keywords:
                        key_lower = key.lower()
                        pattern = re.compile(re.escape(key_lower))
                        for match in pattern.finditer(content_lower):
                            start, end = match.span()
                            prev_char = content_lower[start - 1] if start > 0 else ' '
                            next_char = content_lower[end] if end < len(content_lower) else ' '

                            if not prev_char.isalnum() and prev_char != '_' and not next_char.isalnum() and next_char != '_':
                                algo_scores[algo] += 1

                # Find the algorithm with the highest match count
                best_algo = max(algo_scores, key=algo_scores.get)
                if algo_scores[best_algo] > 0:
                    self.predicted_algo = best_algo
                else:
                    self.predicted_algo = "Unknown"

                # Parse file into instructions
                f.seek(0)
                for line_num, line in enumerate(f, 1):
                    stripped = line.strip()
                    # Filter comments
                    if "/*" in stripped: in_multiline_comment = True
                    if in_multiline_comment:
                        if "*/" in stripped: in_multiline_comment = False
                        continue
                    if not stripped or stripped.startswith(("//", "*", "#")):
                        continue
                    if stripped in ["{", "}", "};"]:
                        continue

                    instruction = stripped.split("//")[0].split("/*")[0].strip()
                    if instruction:
                        clean_instructions.append({
                            "code": instruction,
                            "line_no": line_num
                        })
            return clean_instructions
        except Exception as e:
            print(f"Error loading C file: {e}")
            return []