import re


class MapiProParser:
    """
    Hardware-Aware Static Profiler for MSP430FR6989.
    Filters out comments/headers and maps code to Mapi-Pro energy constants.
    """

    def __init__(self):
        # Energy Constants (nJ) - Mapi-Pro Specs
        self.ENERGY_SRAM_READ = 5500.0
        self.ENERGY_SRAM_WRITE = 5600.0
        self.ENERGY_FRAM_READ = 10325.0
        self.ENERGY_FRAM_WRITE = 13125.0
        self.ENERGY_LOGIC_OP = 2100.0

        # Latency (Clock Cycles @ 16MHz)
        self.LATENCY_SRAM = 1
        self.LATENCY_FRAM = 2
        self.LATENCY_LOGIC = 1

    def profile_line(self, line, mapping_config="HYBRID"):
        """
        Analyzes a single line of C code and calculates hardware costs.
        """
        line = line.strip()

        # Guard: If line is empty or just punctuation, it costs 0
        if not line or line in ["{", "}", "(", ")", ";"]:
            return 0.0, 0

        energy = 0.0
        cycles = 0

        # Heuristic Instruction Classification
        is_write = "=" in line and "==" not in line and "!=" not in line
        # Check for variables or buffer access
        is_read = any(var in line for var in ["data[", "crc", "input", "lin[", "ans"])
        is_logic = any(op in line for op in ["^", ">>", "<<", "&", "|", "+", "-", "*", "++"])

        # Mapi-Pro Memory Mapping
        if mapping_config == "SRAM_ONLY":
            mem_read, mem_write, mem_lat = self.ENERGY_SRAM_READ, self.ENERGY_SRAM_WRITE, self.LATENCY_SRAM
        elif mapping_config == "FRAM_ONLY":
            mem_read, mem_write, mem_lat = self.ENERGY_FRAM_READ, self.ENERGY_FRAM_WRITE, self.LATENCY_FRAM
        else:  # HYBRID (Mapi-Pro Strategy)
            # Buffers like 'lin' or 'icrctb' are likely in FRAM
            if any(buf in line for buf in ["lin[", "icrctb[", "rchr["]):
                mem_read, mem_write, mem_lat = self.ENERGY_FRAM_READ, self.ENERGY_FRAM_WRITE, self.LATENCY_FRAM
            else:
                mem_read, mem_write, mem_lat = self.ENERGY_SRAM_READ, self.ENERGY_SRAM_WRITE, self.LATENCY_SRAM

        # Aggregate Costs
        if is_write:
            energy += mem_write
            cycles += mem_lat
        if is_read:
            energy += mem_read
            cycles += mem_lat
        if is_logic:
            energy += self.ENERGY_LOGIC_OP
            cycles += self.LATENCY_LOGIC

        return energy, cycles

    def load_c_file(self, file_path):
        """
        Extracts executable C instructions and maps them to their
        original source file line numbers.
        """
        clean_instructions = []
        in_multiline_comment = False

        try:
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):  # Start at 1 for editor parity
                    stripped = line.strip()

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
        except FileNotFoundError:
            return []

    def get_checkpoint_cost(self, state_size_bytes=32):
        energy = state_size_bytes * self.ENERGY_FRAM_WRITE
        cycles = state_size_bytes * self.LATENCY_FRAM
        return energy, cycles

    def is_block_boundary(self, line):
        boundaries = ['if', 'else', 'for', 'while', 'return', 'break', 'switch']
        return any(b in line for b in boundaries)