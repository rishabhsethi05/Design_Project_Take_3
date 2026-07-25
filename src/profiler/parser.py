import re


class MapiProParser:
    """
    Hardware-Aware Static Profiler for MSP430FR6989 (FRAM-based).
    Maps C code to physical energy/cycle costs for MAPI-PRO simulation.
    Supports 13+ embedded benchmarks including AES, FFT, SHA256, FIR, MatMult, and BS.
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

        # Benchmark Identification Signatures
        self.signatures = {
            "CRC-16": ["crc", "icrc", "poly", "cword"],
            "QuickSort": ["quicksort", "partition", "pivot", "istack", "jstack", "nstack"],
            "Dijkstra": ["dijkstra", "minnd", "rgnnodes"],
            "BubbleSort": ["bubblesort", "numelems"],
            "Cubic": ["solvecubic", "r2_q3"],
            "InsertSort": ["insertsort"],
            "MergeSort": ["mergesort", "mergesortr", "binarylast", "range_length"],
            "Prime": ["prime", "divides", "even(", "swap("],
            "Hash": ["sglib", "hashed", "htab", "ilist"],
            "StringSearch": ["stringsearch1", "prep1", "exec1", "prep2", "exec2"],
            "Recursion": ["anka(", "kalle("],
            "StrStr": ["strstr", "phaystack", "pneedle", "rhaystack", "rneedle", "foundneedle"],
            "WikiSort": ["wikisort", "wikimerge", "blockswap", "floorpoweroftwo"],
            "Huffman": ["huffbench", "compdecomp", "heap_adjust"],
            "FibCall": ["fibcall", "fnew", "fold", "832040", "apsim_loop"],
            "MatMult": ["matmult", "multiply", "arraya", "arrayb", "resultarray", "upperlimit"],
            "AES": ["cipher", "subbytes", "shiftrows", "mixcolumns", "addroundkey", "keyexpansion"],
            "FFT": ["fft_compute", "realdata", "imagdata", "sintable", "reverse_bits"],
            "SHA256": ["sha256_transform", "sha256_init", "sha256_update", "k_sha", "data_buf"],
            "FIR": ["fir_filter", "fir_coeffs", "fir_input", "fir_output"],
            "DCT": ["dct_compute", "dct_coeffs", "dct_data", "dct_result"],
            "BinarySearch": ["binary_search", "data_array", "search_targets"],
            "Select": ["quick_select", "backup_arr"],
            "Count": ["count_non_negative", "InitSeed", "Array["],
            "PicoJPEG": ["pjpeg", "pjpeg_decode_init", "pjpeg_decode_mcu", "idct", "mcu", "picojpeg"],
        }

        # Non-Volatile FRAM Mapped Arrays and Memory Buffers
        self.fram_buffers = [
            # Standard/Legacy Benchmarks
            "data[", "arr[", "input", "lin[", "rgnNodes", "array[", "x[", "a[", "array1[", "buffer[",
            "memcpy", "htab[", "heap[", "malloc_beebs", "buf[", "search[", "strlen", "text", "substr",
            "phaystack", "pneedle", "rhaystack", "rneedle", "cache[", "wikimerge", "AdjMatrix",
            "freq[", "link[", "code[", "clen[", "heap2[", "outc[", "comp[", "cptr", "dptr",
            "heap_ptr", "heap_end",
            # MatMult
            "ArrayA[", "ArrayB[", "ResultArray[", "Res[", "A[", "B[",
            # AES-128
            "RoundKey[", "sbox[", "Rcon[",
            # FFT
            "RealData[", "ImagData[", "SinTable[",
            # SHA-256
            "K_sha[", "data_buf[", "test_payload[",
            # FIR Filter
            "fir_coeffs[", "fir_input[", "fir_output[",
            "dct_data[", "dct_result[", "dct_coeffs[",
            # Binary Search
            "data_array[", "search_targets[",
            # Select
            "arr[", "backup_arr["
            # Count
            "Array["
            # Prime (Authentic BEEBS)
            "x = ", "y = ", "result ="
            "gCoeffBuf", "gMCUBufR", "gMCUBufG", "gMCUBufB", "HuffTable", "image_info"
        ]

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

        # Check for data-heavy buffer accesses
        is_read = any(keyword in line for keyword in self.fram_buffers)

        # Logic and arithmetic operations
        is_logic = any(op in line for op in
                       ["^", ">>", "<<", "&", "|", "+", "-", "*", "++", "--", "%", "pow(", "acos(", "sqrt(", "cos(",
                        "fabs("])

        # MAPI-PRO Hybrid Memory Mapping Logic
        # Heavy arrays and static buffers map to FRAM; local scalars map to SRAM
        if any(buf in line for buf in self.fram_buffers):
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
        Normalized so that the baseline 32-byte cost remains anchored to original physics.
        """
        # Base calibration: 13500 nJ was calibrated for the whole 32-byte block
        PER_BYTE_ENERGY = self.ENERGY_FRAM_WRITE / 32.0

        # FRAM latency is low, assume 1 cycle per 2-byte word write
        PER_BYTE_LATENCY = 0.5

        energy = state_size_bytes * PER_BYTE_ENERGY
        cycles = int(state_size_bytes * PER_BYTE_LATENCY)

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