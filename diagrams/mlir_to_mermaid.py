#!/usr/bin/env python3
"""
MLIR to Mermaid Diagram Generator

Automatically generates interactive Mermaid diagrams from MLIR d2m.generic operations.

Features:
- Automatically extracts threads, circular buffers, and semaphores from MLIR
- Infers producer/consumer relationships from d2m.reserve and d2m.wait operations
- Generates three diagram types:
  1. Dataflow: Shows thread communication via circular buffers
  2. Sequence: Shows temporal execution timeline
  3. Architecture: Shows grid topology and core layout

Usage:
    python mlir_to_mermaid.py <mlir_file> [options]

Options:
    --diagram-type {dataflow,sequence,architecture,all}
    --output <output_file.md>
    --interactive  (ask user for preferences)

Examples:
    python mlir_to_mermaid.py matmul.mlir
    python mlir_to_mermaid.py program.mlir --diagram-type dataflow
    python mlir_to_mermaid.py program.mlir --interactive
"""

import re
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path


@dataclass
class CircularBuffer:
    """Represents a circular buffer in the MLIR code."""

    name: str
    index: int
    tensor_type: str
    producer: Optional[str] = None
    consumers: List[str] = field(default_factory=list)


@dataclass
class Semaphore:
    """Represents a semaphore."""

    name: str
    index: int
    purpose: str = ""


@dataclass
class Thread:
    """Represents a thread (datamovement or compute)."""

    name: str
    thread_type: str  # 'datamovement' or 'compute'
    index: int
    operations: List[str] = field(default_factory=list)
    cb_accesses: List[str] = field(default_factory=list)
    cbs_produced: List[int] = field(
        default_factory=list
    )  # CB indices this thread produces
    cbs_consumed: List[int] = field(
        default_factory=list
    )  # CB indices this thread consumes


@dataclass
class D2MGeneric:
    """Represents a d2m.generic operation."""

    grid: Tuple[int, int]
    threads: List[Thread]
    circular_buffers: List[CircularBuffer]
    semaphores: List[Semaphore]
    inputs: List[str]
    outputs: List[str]


class MLIRParser:
    """Parse MLIR to extract D2M information."""

    def __init__(self, mlir_text: str):
        self.mlir_text = mlir_text
        self.generic_op = None

    def parse(self) -> Optional[D2MGeneric]:
        """Parse the MLIR and extract d2m.generic information."""
        # Extract grid
        grid_match = re.search(r"grid = #ttcore\.grid<(\d+)x(\d+)>", self.mlir_text)
        if not grid_match:
            return None
        grid = (int(grid_match.group(1)), int(grid_match.group(2)))

        # Extract threads
        threads_match = re.search(r"threads = \[(.*?)\]", self.mlir_text)
        if not threads_match:
            return None

        thread_list = []
        thread_strs = threads_match.group(1).split(",")
        dm_count = 0
        for i, thread_str in enumerate(thread_strs):
            if "datamovement" in thread_str:
                thread_list.append(Thread(f"DM{dm_count}", "datamovement", i))
                dm_count += 1
            elif "compute" in thread_str:
                thread_list.append(Thread(f"Compute", "compute", i))

        # Extract circular buffers from block arguments (deduplicate by index)
        cb_map = {}
        cb_pattern = r"%cb(\d+): !d2m\.cb<(.+?)>"
        for match in re.finditer(cb_pattern, self.mlir_text):
            cb_idx = int(match.group(1))
            cb_type = match.group(2)
            if cb_idx not in cb_map:
                cb_map[cb_idx] = CircularBuffer(f"CB{cb_idx}", cb_idx, cb_type)
        cbs = [cb_map[idx] for idx in sorted(cb_map.keys())]

        # Extract semaphores (deduplicate by index)
        sem_map = {}
        sem_pattern = r"%sem(\d+): !d2m\.semaphore"
        for match in re.finditer(sem_pattern, self.mlir_text):
            sem_idx = int(match.group(1))
            if sem_idx not in sem_map:
                sem_map[sem_idx] = Semaphore(f"sem{sem_idx}", sem_idx)
        sems = [sem_map[idx] for idx in sorted(sem_map.keys())]

        # Parse thread regions
        self._parse_thread_regions(thread_list)

        # Infer CB producers/consumers
        self._infer_cb_relationships(thread_list, cbs)

        return D2MGeneric(
            grid=grid,
            threads=thread_list,
            circular_buffers=cbs,
            semaphores=sems,
            inputs=[],
            outputs=[],
        )

    def _parse_thread_regions(self, threads: List[Thread]):
        """Parse operations within each thread region and track CB usage."""
        # Find thread regions - match region name with number
        # Regions end with either "}, {" (next region) or just "}" (last region)
        region_pattern = (
            r"\^(datamovement|compute)(\d+)\(.*?\):\s*\n(.*?)(?=\s*\}, \{|\s*\}\s*$)"
        )

        # Build a map from region name to thread
        region_map = {}
        dm_count = 0
        compute_count = 0
        for thread in threads:
            if thread.thread_type == "datamovement":
                region_map[f"datamovement{dm_count}"] = thread
                dm_count += 1
            elif thread.thread_type == "compute":
                region_map[f"compute{compute_count}"] = thread
                compute_count += 1

        for match in re.finditer(region_pattern, self.mlir_text, re.DOTALL):
            thread_type = match.group(1)
            thread_idx = match.group(2)
            region_body = match.group(3)
            region_name = f"{thread_type}{thread_idx}"

            # Find matching thread
            if region_name in region_map:
                thread = region_map[region_name]

                # Extract key operations
                ops = []
                if "d2m.dma" in region_body:
                    ops.append("DMA transfer")
                if "d2m.reserve" in region_body:
                    ops.append("reserve CB")
                if "d2m.wait" in region_body:
                    ops.append("wait CB")
                if "semaphore_wait" in region_body:
                    ops.append("semaphore_wait")
                if "semaphore_set" in region_body:
                    ops.append("semaphore_set")
                if "semaphore_inc" in region_body:
                    ops.append("semaphore_inc")
                if "mcast" in region_body:
                    ops.append("multicast")
                if "tile_matmul" in region_body:
                    ops.append("tile_matmul")
                if "d2m.store" in region_body:
                    ops.append("store to CB")

                thread.operations = ops

                # Track CB usage: reserve = produce, wait = consume
                thread.cbs_produced = []
                thread.cbs_consumed = []

                # Find CBs this thread reserves (produces)
                for reserve_match in re.finditer(r"d2m\.reserve %cb(\d+)", region_body):
                    cb_idx = int(reserve_match.group(1))
                    if cb_idx not in thread.cbs_produced:
                        thread.cbs_produced.append(cb_idx)

                # Find CBs this thread waits on (consumes)
                for wait_match in re.finditer(r"d2m\.wait %cb(\d+)", region_body):
                    cb_idx = int(wait_match.group(1))
                    if cb_idx not in thread.cbs_consumed:
                        thread.cbs_consumed.append(cb_idx)

    def _infer_cb_relationships(self, threads: List[Thread], cbs: List[CircularBuffer]):
        """Infer which threads produce/consume which CBs based on actual usage."""
        # Use the parsed CB usage from thread regions
        for thread in threads:
            # Set producer for CBs this thread produces
            for cb_idx in thread.cbs_produced:
                if cb_idx < len(cbs):
                    cbs[cb_idx].producer = thread.name

            # Add to consumers for CBs this thread consumes
            for cb_idx in thread.cbs_consumed:
                if cb_idx < len(cbs):
                    if thread.name not in cbs[cb_idx].consumers:
                        cbs[cb_idx].consumers.append(thread.name)

        # For CBs with no consumers, mark as output
        for cb in cbs:
            if not cb.consumers:
                cb.consumers = ["Output"]


class MermaidGenerator:
    """Generate Mermaid diagrams from parsed MLIR."""

    def __init__(self, generic_op: D2MGeneric, core_id: Tuple[int, int] = (0, 0)):
        self.op = generic_op
        self.core_id = core_id

    def generate_dataflow(self) -> str:
        """Generate dataflow diagram."""
        lines = [
            "```mermaid",
            "---",
            f'title: "Detailed Tile Flow - Single Iteration in Core[{self.core_id[0]},{self.core_id[1]}]"',
            "---",
            '%%{ init: { "theme": "base", "themeVariables": { "primaryColor": "#e8f5e9", "primaryTextColor": "#000", "primaryBorderColor": "#2e7d32", "lineColor": "#666", "secondaryColor": "#f3e5f5", "tertiaryColor": "#fff9c4", "clusterBkg": "#f5f5f5", "clusterBorder": "#666", "edgeLabelBackground": "#ffffff", "fontSize": "18px" }, "flowchart": { "markdownAutoWrap": false, "wrappingWidth": 9999, "nodeSpacing": 60, "rankSpacing": 60 } } }%%',
            "flowchart TD",
            '    subgraph iteration["Processing Iteration"]',
            "        direction TB",
            "        ",
            '        subgraph dram["💾 DRAM Global Memory"]',
            '            input_tiles["Input Tiles<br/>From global memory"]',
            "        end",
            "        ",
            f'        subgraph core_l1["Core[{self.core_id[0]},{self.core_id[1]}]_L1_Memory_And_Compute"]',
            "            direction TB",
            "            ",
            f'            subgraph threads["⚡ {len(self.op.threads)} Concurrent Threads"]',
            "                direction LR",
            "                ",
        ]

        # Generate thread subgraphs
        for thread in self.op.threads:
            icon = "🔄" if thread.thread_type == "datamovement" else "⚙️"
            lines.append(
                f'                subgraph {thread.name.lower()}_flow["{icon} {thread.name} Thread Flow"]'
            )
            lines.append(f"                    direction TB")

            for i, op in enumerate(thread.operations, 1):
                op_id = f"{thread.name.lower()}_{i}"
                lines.append(f'                    {op_id}["<b>{i}️⃣ {op}</b>"]')
                # Add connections between steps
                if i > 1:
                    prev_id = f"{thread.name.lower()}_{i-1}"
                    lines.append(f"                    {prev_id} --> {op_id}")

            lines.append("                end")
            lines.append("                ")

        lines.append("            end")
        lines.append("            ")

        # Generate circular buffers
        lines.append('            subgraph cbs["📦 L1 Circular Buffers"]')
        lines.append("                direction LR")

        for cb in self.op.circular_buffers:
            producer_str = (
                f"Producer: {cb.producer}" if cb.producer else "Producer: Unknown"
            )
            consumer_str = (
                f"Consumer: {', '.join(cb.consumers)}"
                if cb.consumers
                else "Consumer: Unknown"
            )
            lines.append(
                f'                {cb.name.lower()}["{cb.name}<br/>Multiple tile slots<br/>{producer_str}<br/>{consumer_str}"]'
            )

        lines.append("            end")
        lines.append("            ")

        # Generate semaphores (vertical list for narrower box)
        lines.append('            subgraph sync["🚦 Synchronization"]')
        lines.append("                direction TB")
        # Put each unique semaphore on its own line for a narrower, taller box
        unique_sems = sorted(set(s.name for s in self.op.semaphores))
        sem_lines = "<br/>".join([f"• {name}" for name in unique_sems])
        lines.append(
            f'                sems["<b>Semaphores:</b><br/>{sem_lines}<br/><br/>Flow control and<br/>synchronization"]'
        )
        lines.append("            end")
        lines.append("        end")
        lines.append("    end")
        lines.append("    ")

        # Add connections
        lines.append("    %% Data movement connections")

        # Connect input to DM threads that have DMA operations
        for thread in self.op.threads:
            if (
                thread.thread_type == "datamovement"
                and "DMA transfer" in thread.operations
            ):
                if len(thread.operations) > 0:
                    first_op = f"{thread.name.lower()}_1"
                    lines.append(f'    input_tiles ==>|"DMA transfer"| {first_op}')

        lines.append("    ")

        # Thread to CB connections (producers write to CBs)
        for thread in self.op.threads:
            for cb_idx in thread.cbs_produced:
                cb_name = f"cb{cb_idx}"
                # Find the reserve/write operation
                for op_idx, op in enumerate(thread.operations, 1):
                    if "reserve" in op.lower() or "store" in op.lower():
                        lines.append(
                            f'    {thread.name.lower()}_{op_idx} ==>|"Write tiles"| {cb_name}'
                        )
                        break

        # CB to thread connections (consumers read from CBs)
        for thread in self.op.threads:
            for cb_idx in thread.cbs_consumed:
                cb_name = f"cb{cb_idx}"
                # Find a relevant consume operation (wait, compute, etc.)
                for op_idx, op in enumerate(thread.operations, 1):
                    if any(
                        keyword in op.lower()
                        for keyword in ["wait", "matmul", "compute"]
                    ):
                        lines.append(
                            f'    {cb_name} ==>|"Read tiles"| {thread.name.lower()}_{op_idx}'
                        )
                        break

        # Semaphore connections (to threads that use semaphores)
        lines.append("    ")
        lines.append("    %% Synchronization")
        for thread in self.op.threads:
            # Check if thread uses semaphores
            if any("semaphore" in op.lower() for op in thread.operations):
                # Find an operation that uses semaphores
                for op_idx, op in enumerate(thread.operations, 1):
                    if "semaphore" in op.lower():
                        lines.append(
                            f'    sems -.->|"coordinate"| {thread.name.lower()}_{op_idx}'
                        )
                        break
        lines.append("    ")

        # Styling - dynamically style each thread flow box
        # Note: Removing explicit text colors to allow theme adaptation for light/dark mode
        lines.append("    %% Styling")
        lines.append("    style dram fill:#ffebee,stroke:#c62828,stroke-width:2px")
        lines.append("    style core_l1 fill:#f5f5f5,stroke:#424242,stroke-width:4px")
        lines.append("    style threads fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px")
        lines.append("    style iteration fill:#ffffff,stroke:#666,stroke-width:2px")

        # Style thread flows with darker borders for distinction
        for thread in self.op.threads:
            flow_id = f"{thread.name.lower()}_flow"
            if thread.thread_type == "datamovement":
                # Pink/magenta theme for datamovement threads
                lines.append(
                    f"    style {flow_id} fill:#f3e5f5,stroke:#7b1fa2,stroke-width:3px"
                )
            else:
                # Green theme for compute threads
                lines.append(
                    f"    style {flow_id} fill:#e8f5e9,stroke:#388e3c,stroke-width:3px"
                )

        lines.append("    style cbs fill:#fff9c4,stroke:#f57f17,stroke-width:3px")
        lines.append("    style sync fill:#ffccbc,stroke:#d84315,stroke-width:2px")
        lines.append("```")

        return "\n".join(lines)

    def generate_sequence(self) -> str:
        """Generate sequence diagram."""
        lines = [
            "```mermaid",
            '%%{ init: { "theme": "base", "themeVariables": { "fontSize": "18px", "actorBkg": "#e8f5e9", "actorBorder": "#2e7d32", "actorLineColor": "#666", "labelBoxBkgColor": "#e3f2fd", "labelBoxBorderColor": "#1976d2", "noteBkgColor": "#fff9c4", "noteBorderColor": "#f57f17", "activationBkgColor": "#e8f5e9", "activationBorderColor": "#2e7d32", "signalColor": "#666", "signalTextColor": "#666" }, "sequence": { "messageAlign": "center", "mirrorActors": false } } }%%',
            "sequenceDiagram",
            "    participant DRAM",
        ]

        # Add participants for each thread and CB
        for thread in self.op.threads:
            lines.append(f"    participant {thread.name}")

        for cb in self.op.circular_buffers[:3]:  # Show first 3 CBs
            lines.append(f"    participant {cb.name}")

        lines.append("    ")
        lines.append(
            "    Note over DRAM,"
            + self.op.circular_buffers[-1].name
            + ": Iteration k=0"
        )
        lines.append("    ")

        # Generate sequences for each thread based on their operations
        for thread_idx, thread in enumerate(self.op.threads):
            is_parallel = thread_idx > 0
            if is_parallel:
                lines.append(f"    par Parallel {thread.name} Thread")

            # Generate sequence based on actual operations
            indent = "        " if is_parallel else "    "

            # Start with DMA if it's a datamovement thread
            if (
                thread.thread_type == "datamovement"
                and "DMA transfer" in thread.operations
            ):
                lines.append(f"{indent}DRAM->>+{thread.name}: DMA load tiles")

            # Show CB writes (reserve/store operations)
            for cb_idx in thread.cbs_produced:
                if cb_idx < len(self.op.circular_buffers):
                    cb_name = self.op.circular_buffers[cb_idx].name
                    lines.append(f"{indent}{thread.name}->>{cb_name}: reserve & write")

            # Show CB reads (wait operations)
            for cb_idx in thread.cbs_consumed:
                if cb_idx < len(self.op.circular_buffers):
                    cb_name = self.op.circular_buffers[cb_idx].name
                    lines.append(f"{indent}{cb_name}->>+{thread.name}: wait & read")

            # Show key operations
            for op in thread.operations:
                if "semaphore" in op.lower():
                    lines.append(f"{indent}{thread.name}->>{thread.name}: {op}")
                elif "multicast" in op.lower():
                    lines.append(
                        f"{indent}{thread.name}->>{thread.name}: multicast to other cores"
                    )
                elif "matmul" in op.lower():
                    lines.append(
                        f"{indent}{thread.name}->>{thread.name}: compute (matmul)"
                    )

            # End sequence
            if thread.thread_type == "datamovement":
                lines.append(f"{indent}{thread.name}->>-DRAM: Done")

            if is_parallel:
                lines.append("    end")
            lines.append("    ")

        lines.extend(
            [
                f"    Note over DRAM,{self.op.circular_buffers[-1].name}: All {len(self.op.threads)} threads execute concurrently",
                "```",
            ]
        )

        return "\n".join(lines)

    def generate_architecture(self) -> str:
        """Generate grid architecture diagram."""
        lines = [
            "```mermaid",
            f"---",
            f'title: "D2M Grid Architecture - {self.op.grid[0]}x{self.op.grid[1]} Cores"',
            f"---",
            '%%{ init: { "theme": "base", "themeVariables": { "primaryColor": "#e8f5e9", "primaryTextColor": "#000", "primaryBorderColor": "#2e7d32", "lineColor": "#666", "secondaryColor": "#f3e5f5", "tertiaryColor": "#fff9c4", "clusterBkg": "#f5f5f5", "clusterBorder": "#666", "edgeLabelBackground": "#ffffff", "fontSize": "18px" }, "flowchart": { "markdownAutoWrap": false, "wrappingWidth": 9999, "nodeSpacing": 60, "rankSpacing": 60 } } }%%',
            "flowchart LR",
            '    subgraph grid["⚡ Core Grid"]',
            "        direction TB",
            "        ",
        ]

        # Generate cores
        for row in range(self.op.grid[0]):
            for col in range(self.op.grid[1]):
                core_id = f"C{row}{col}"
                thread_info = "<br/>───────<br/>"
                for thread in self.op.threads:
                    thread_info += (
                        f'{thread.name}: {", ".join(thread.operations[:2])}<br/>'
                    )

                lines.append(f'        {core_id}["Core[{row},{col}]{thread_info}"]')

        lines.extend(
            [
                "    end",
                "    ",
                '    subgraph memory["💾 Global Memory"]',
                '        INPUT["Input Data"]',
                "    end",
                "    ",
            ]
        )

        # Add connections
        lines.extend(
            [
                '    INPUT ==>|"DMA"| C00',
                "    ",
            ]
        )

        # Add multicast connections from C00 to all other cores
        if self.op.grid[0] > 1 or self.op.grid[1] > 1:
            for row in range(self.op.grid[0]):
                for col in range(self.op.grid[1]):
                    if row == 0 and col == 0:
                        continue  # Skip C00 itself
                    core_id = f"C{row}{col}"
                    lines.append(f'    C00 ==>|"Multicast"| {core_id}')
            lines.append("    ")

        lines.extend(
            [
                "    style C00 fill:#d4edda,stroke:#28a745,stroke-width:3px",
                "    style memory fill:#ffe1e1",
                "```",
            ]
        )

        return "\n".join(lines)


def interactive_menu() -> Tuple[str, str]:
    """Interactive menu for diagram selection."""
    print("\n" + "=" * 60)
    print("MLIR to Mermaid Diagram Generator")
    print("=" * 60)

    print("\nAvailable diagram types:")
    print("  1. Dataflow - Shows thread communication via circular buffers")
    print("  2. Sequence - Shows temporal execution timeline")
    print("  3. Architecture - Shows grid topology and core layout")
    print("  4. All - Generate all diagram types")

    choice = input("\nSelect diagram type (1-4): ").strip()

    diagram_map = {"1": "dataflow", "2": "sequence", "3": "architecture", "4": "all"}

    diagram_type = diagram_map.get(choice, "dataflow")

    output_file = input("\nOutput file (default: diagrams.md): ").strip()
    if not output_file:
        output_file = "diagrams.md"

    return diagram_type, output_file


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python mlir_to_mermaid.py <mlir_file> [--interactive]")
        sys.exit(1)

    mlir_file = sys.argv[1]

    if not Path(mlir_file).exists():
        print(f"Error: File not found: {mlir_file}")
        sys.exit(1)

    # Read MLIR file
    with open(mlir_file, "r") as f:
        mlir_text = f.read()

    # Parse MLIR
    parser = MLIRParser(mlir_text)
    generic_op = parser.parse()

    if not generic_op:
        print("Error: Could not parse d2m.generic operation from MLIR file")
        sys.exit(1)

    print(f"✓ Parsed d2m.generic with {generic_op.grid[0]}x{generic_op.grid[1]} grid")
    print(f"✓ Found {len(generic_op.threads)} threads")
    print(f"✓ Found {len(generic_op.circular_buffers)} circular buffers")

    # Determine diagram type and output
    if "--interactive" in sys.argv or "-i" in sys.argv:
        diagram_type, output_file = interactive_menu()
    else:
        diagram_type = "all"
        output_file = mlir_file.replace(".mlir", "_diagrams.md")

    # Generate diagrams
    generator = MermaidGenerator(generic_op)

    diagrams = []

    if diagram_type in ["dataflow", "all"]:
        print("\n✓ Generating dataflow diagram...")
        diagrams.append(("Dataflow Diagram", generator.generate_dataflow()))

    if diagram_type in ["sequence", "all"]:
        print("✓ Generating sequence diagram...")
        diagrams.append(("Sequence Diagram", generator.generate_sequence()))

    if diagram_type in ["architecture", "all"]:
        print("✓ Generating architecture diagram...")
        diagrams.append(("Architecture Diagram", generator.generate_architecture()))

    # Write markdown output
    with open(output_file, "w") as f:
        f.write(f"# MLIR Diagrams: {Path(mlir_file).name}\n\n")
        f.write(f"Generated from: `{mlir_file}`\n\n")
        f.write(
            "> **Note**: To view these diagrams in Cursor/VSCode, install the 'Markdown Preview Mermaid Support' extension.\n"
        )
        f.write("> Alternatively, open the accompanying `.html` file in a browser.\n\n")
        f.write("---\n\n")

        for title, diagram in diagrams:
            f.write(f"## {title}\n\n")
            f.write(diagram)
            f.write("\n\n---\n\n")

    # Also write HTML output for easy browser viewing
    html_file = output_file.replace(".md", ".html")
    html_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>MLIR Diagrams</title>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
    </script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{ color: #333; border-bottom: 3px solid #007acc; padding-bottom: 10px; font-size: 28px; }}
        h2 {{ color: #555; margin-top: 40px; font-size: 24px; }}
        .mermaid {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin: 20px 0; }}
        .info {{ background: #e3f2fd; padding: 15px; border-left: 4px solid #2196f3; margin: 20px 0; font-size: 16px; }}
        /* Increase Mermaid diagram font sizes */
        .mermaid svg {{ font-size: 16px !important; }}
        .mermaid .node rect, .mermaid .node circle, .mermaid .node polygon {{ font-size: 16px !important; }}
        .mermaid .label {{ font-size: 16px !important; }}
        .mermaid text {{ font-size: 16px !important; }}
        /* Make subgraph/cluster labels larger, bold, with more padding */
        .mermaid .cluster-label {{ font-size: 18px !important; font-weight: 700 !important; padding: 14px !important; margin: 8px !important; white-space: nowrap !important; }}
        .mermaid .cluster text {{ font-size: 18px !important; font-weight: 700 !important; white-space: nowrap !important; }}
        .mermaid .cluster rect {{ padding: 12px !important; }}
        .mermaid .cluster .nodeLabel {{ white-space: nowrap !important; }}
    </style>
</head>
<body>
    <h1>MLIR Diagrams: {0}</h1>
    <div class="info">
        <strong>Generated from:</strong> <code>{1}</code>
    </div>
"""

    with open(html_file, "w") as f:
        f.write(html_template.format(Path(mlir_file).name, mlir_file))

        for title, diagram in diagrams:
            # Extract just the mermaid code (remove the ```mermaid and ``` wrapper)
            mermaid_code = diagram.replace("```mermaid\n", "").replace("\n```", "")
            f.write(f"    <h2>{title}</h2>\n")
            f.write(f'    <div class="mermaid">\n{mermaid_code}\n    </div>\n\n')

        f.write(
            """
</body>
</html>
"""
        )

    print(f"\n✓ Diagrams written to:")
    print(f"  - Markdown: {output_file}")
    print(f"  - HTML: {html_file}")
    print(f"\nView options:")
    print(f"  1. Open {html_file} in your browser (recommended)")
    print(f"  2. Install 'Markdown Preview Mermaid Support' extension in Cursor/VSCode")
    print(f"  3. Push to GitHub/GitLab (renders Mermaid automatically)")
    print(f"  4. Paste diagram code into https://mermaid.live")


if __name__ == "__main__":
    main()
