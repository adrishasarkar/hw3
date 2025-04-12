#!/usr/bin/env python3
import re
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def parse_log_file(filename):
    """
    Parse the log file to extract insertion and assembly times along with corresponding nodes and ranks per node.
    Returns:
      - insert_dict: {(nodes, ranks per node): insertion time}
      - assemble_dict: {(nodes, ranks per node): assembly time}
      - sorted list of unique nodes
      - sorted list of unique ranks per node
    """
    insert_dict = {}
    assemble_dict = {}
    nodes_set = set()
    ranks_set = set()

    # Regular expressions for matching the output lines.
    insert_regex = re.compile(
        r"Finished inserting in ([0-9.]+).*ranks per node:\s*(\d+),\s*nodes:\s*(\d+)"
    )
    assemble_regex = re.compile(
        r"Assembled in ([0-9.]+).*ranks per node:\s*(\d+),\s*nodes:\s*(\d+)"
    )

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            match = insert_regex.search(line)
            if match:
                time_val, ranks, nodes = match.groups()
                time_val = float(time_val)
                ranks = int(ranks)
                nodes = int(nodes)
                insert_dict[(nodes, ranks)] = time_val
                nodes_set.add(nodes)
                ranks_set.add(ranks)
                continue

            match = assemble_regex.search(line)
            if match:
                time_val, ranks, nodes = match.groups()
                time_val = float(time_val)
                ranks = int(ranks)
                nodes = int(nodes)
                assemble_dict[(nodes, ranks)] = time_val
                nodes_set.add(nodes)
                ranks_set.add(ranks)

    return insert_dict, assemble_dict, sorted(nodes_set), sorted(ranks_set)

def build_dataframe(data_dict, nodes, ranks):
    """
    Build a 2D pandas DataFrame from the provided dictionary.
    The DataFrame has rows as nodes and columns as ranks per node.
    Missing entries will remain as NaN.
    """
    df = pd.DataFrame(index=nodes, columns=ranks)
    for n in nodes:
        for r in ranks:
            df.loc[n, r] = data_dict.get((n, r), None)
    return df

def plot_insertion_and_assembly(insert_df, assembly_df, nodes, ranks):
    """
    Create a log-log plot with:
      - x-axis: total number of tasks (nodes * ranks per node)
      - y-axis: time in seconds (log scale)
      
    Both insertion and assembly times are plotted on the same figure.
    For each fixed number of nodes, insertion times (circle markers, solid lines)
    and assembly times (square markers, dashed lines) are connected by a line.
    Additionally, for each metric an ideal scaling line is drawn as:
         T = a / (total_tasks)
    where a is the measured time for (1 node, 1 rank per node).
    
    The figure is saved as a JPEG file with 300 dpi.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    # Determine total_tasks range for the ideal curves.
    all_total_tasks = []
    for node in nodes:
        for rank in ranks:
            total_tasks = node * rank
            if (pd.notna(insert_df.loc[node, rank]) or pd.notna(assembly_df.loc[node, rank])):
                all_total_tasks.append(total_tasks)
    if all_total_tasks:
        x_min = min(all_total_tasks)
        x_max = max(all_total_tasks)
    else:
        x_min, x_max = 1, 1
    
    # Plot measured insertion and assembly times for each node.
    for node in nodes:
        x_insertion = []
        y_insertion = []
        x_assembly = []
        y_assembly = []
        for rank in ranks:
            total_tasks = node * rank
            time_ins = insert_df.loc[node, rank]
            time_ass = assembly_df.loc[node, rank]
            if pd.notna(time_ins):
                x_insertion.append(total_tasks)
                y_insertion.append(time_ins)
            if pd.notna(time_ass):
                x_assembly.append(total_tasks)
                y_assembly.append(time_ass)
        
        if x_insertion:
            ax.plot(x_insertion, y_insertion, marker='o', linestyle='-',
                    label=f"Nodes = {node} Insertion")
        if x_assembly:
            ax.plot(x_assembly, y_assembly, marker='s', linestyle='-',
                    label=f"Nodes = {node} Assembly")
    
    # Plot ideal scaling lines using (1,1) measurement.
    x_vals = np.logspace(np.log10(x_min), np.log10(x_max), num=100)
    try:
        a_insertion = float(insert_df.loc[1, 1])
    except (KeyError, ValueError, TypeError):
        a_insertion = None
    if a_insertion is not None and not pd.isna(a_insertion):
        ideal_insertion = a_insertion / x_vals
        ax.plot(x_vals, ideal_insertion, linestyle=':', color='black',
                label=r"Ideal Insertion Scaling ($x^{-1}$)")
    
    try:
        a_assembly = float(assembly_df.loc[1, 1])
    except (KeyError, ValueError, TypeError):
        a_assembly = None
    if a_assembly is not None and not pd.isna(a_assembly):
        ideal_assembly = a_assembly / x_vals
        ax.plot(x_vals, ideal_assembly, linestyle='--', color='black',
                label=r"Ideal Assembly Scaling ($x^{-1}$)")
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Total Ranks (nodes × ranks per node)")
    ax.set_ylabel("Time (s)")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, which="both")
    fig.tight_layout()
    fig.savefig("insertion_assembly_times.jpg", dpi=300)
    plt.show()

def plot_time_vs_nodes(insert_df, assembly_df, nodes, target_ranks=[60, 64]):
    """
    Create a log-log plot with:
      - x-axis: number of nodes
      - y-axis: time in seconds (log scale)
    
    For each fixed number of ranks per node (e.g., 60 and 64), the measured
    insertion times (circle markers, solid lines) and assembly times (square markers, dashed lines)
    are plotted. Additionally, an ideal scaling line is drawn assuming ideal strong scaling:
         T = T(1, r) / (nodes)
    where T(1, r) is the measured time for (1 node, fixed r).
    
    Only one legend entry for Ideal Insertion Scaling and one for Ideal Assembly Scaling is added.
    
    The figure is saved as a JPEG file with 300 dpi.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    # Flags to ensure only one ideal scaling label is added per metric.
    ideal_insertion_added = False
    ideal_assembly_added = False
    
    for r in target_ranks:
        nodes_ins = []
        times_ins = []
        nodes_ass = []
        times_ass = []
        
        # Gather data for the fixed number of ranks r.
        for node in nodes:
            # Insertion times for fixed rank r.
            if r in insert_df.columns:
                time_ins = insert_df.loc[node, r]
                if pd.notna(time_ins):
                    nodes_ins.append(node)
                    times_ins.append(time_ins)
            # Assembly times for fixed rank r.
            if r in assembly_df.columns:
                time_ass = assembly_df.loc[node, r]
                if pd.notna(time_ass):
                    nodes_ass.append(node)
                    times_ass.append(time_ass)
                    
        # Plot measured data for insertion.
        if nodes_ins:
            ax.plot(nodes_ins, times_ins, marker='o', linestyle='-',
                    label=f"Ranks = {r} Insertion")
        # Plot measured data for assembly.
        if nodes_ass:
            ax.plot(nodes_ass, times_ass, marker='s', linestyle='--',
                    label=f"Ranks = {r} Assembly")
        
        # Plot ideal scaling lines if measurement at 1 node is available.
        if 1 in nodes and (r in insert_df.columns) and pd.notna(insert_df.loc[1, r]):
            a_insertion = insert_df.loc[1, r]
            x_vals = np.logspace(np.log10(min(nodes)), np.log10(max(nodes)), num=100)
            ideal_insertion = a_insertion / x_vals
            lab = "Ideal Insertion Scaling" if not ideal_insertion_added else None
            ideal_insertion_added = True
            ax.plot(x_vals, ideal_insertion, linestyle=':', color='black', label=lab)
        
        if 1 in nodes and (r in assembly_df.columns) and pd.notna(assembly_df.loc[1, r]):
            a_assembly = assembly_df.loc[1, r]
            x_vals = np.logspace(np.log10(min(nodes)), np.log10(max(nodes)), num=100)
            ideal_assembly = a_assembly / x_vals
            lab = "Ideal Assembly Scaling" if not ideal_assembly_added else None
            ideal_assembly_added = True
            ax.plot(x_vals, ideal_assembly, linestyle='-.', color='black', label=lab)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel("Time (s)")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, which="both")
    fig.tight_layout()
    fig.savefig("time_vs_nodes.jpg", dpi=300)
    plt.show()

def plot_time_vs_ranks(insert_df, assembly_df, nodes, ranks):
    """
    Create a log-log plot with:
      - x-axis: number of ranks per node (for node = 1 only)
      - y-axis: time in seconds (log scale)
    
    For node 1, the measured insertion times (circle markers, solid lines) and 
    assembly times (square markers, dashed lines) are plotted versus the number of ranks per node.
    Additionally, ideal scaling curves are added as:
         T = T(1,1) / (ranks)
    for both insertion and assembly.
    
    The figure is saved as a JPEG file with 300 dpi.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    # Only consider node 1.
    if 1 not in nodes:
        print("No data for node 1")
        return

    # Collect measured data for node 1.
    x_ranks_ins = []
    y_ins = []
    x_ranks_ass = []
    y_ass = []
    for r in ranks:
        t_ins = insert_df.loc[1, r]
        t_ass = assembly_df.loc[1, r]
        if pd.notna(t_ins):
            x_ranks_ins.append(r)
            y_ins.append(t_ins)
        if pd.notna(t_ass):
            x_ranks_ass.append(r)
            y_ass.append(t_ass)
    
    # Plot measured insertion and assembly times.
    if x_ranks_ins:
        ax.plot(x_ranks_ins, y_ins, marker='o', linestyle='-', label="Node 1 Insertion")
    if x_ranks_ass:
        ax.plot(x_ranks_ass, y_ass, marker='s', linestyle='--', label="Node 1 Assembly")
    
    # Add ideal scaling lines (T = T(1,1) / r).
    try:
        a_insertion = float(insert_df.loc[1, 1])
    except (KeyError, ValueError, TypeError):
        a_insertion = None
    if a_insertion is not None and not pd.isna(a_insertion):
        x_vals = np.logspace(np.log10(min(ranks)), np.log10(max(ranks)), num=100)
        ideal_insertion = a_insertion / x_vals
        ax.plot(x_vals, ideal_insertion, linestyle=':', color='black',
                label=r"Ideal Insertion Scaling ($x^{-1}$)")
    
    try:
        a_assembly = float(assembly_df.loc[1, 1])
    except (KeyError, ValueError, TypeError):
        a_assembly = None
    if a_assembly is not None and not pd.isna(a_assembly):
        x_vals = np.logspace(np.log10(min(ranks)), np.log10(max(ranks)), num=100)
        ideal_assembly = a_assembly / x_vals
        ax.plot(x_vals, ideal_assembly, linestyle='-.', color='black',
                label=r"Ideal Assembly Scaling ($x^{-1}$)")
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Ranks per Node (Node = 1)")
    ax.set_ylabel("Time (s)")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, which="both")
    fig.tight_layout()
    fig.savefig("time_vs_ranks_node1.jpg", dpi=300)
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_log.py <log_file.txt>")
        sys.exit(1)

    log_file = sys.argv[1]
    
    # Parse the log file.
    insert_dict, assemble_dict, nodes, ranks = parse_log_file(log_file)
    
    # Build DataFrames for insertion and assembly times.
    insert_df = build_dataframe(insert_dict, nodes, ranks)
    assembly_df = build_dataframe(assemble_dict, nodes, ranks)
    
    print("Insertion Times DataFrame (rows: nodes, columns: ranks per node):")
    print(insert_df)
    print("\nAssembly Times DataFrame (rows: nodes, columns: ranks per node):")
    print(assembly_df)
    
    # Plot measured and ideal scaling vs. total tasks.
    plot_insertion_and_assembly(insert_df, assembly_df, nodes, ranks)
    
    # Plot time vs. number of nodes for fixed ranks 60 and 64.
    plot_time_vs_nodes(insert_df, assembly_df, nodes, target_ranks=[60, 64])
    
    # Plot time vs. number of ranks (only for node 1) with ideal 1/x scaling.
    plot_time_vs_ranks(insert_df, assembly_df, nodes, ranks)

if __name__ == "__main__":
    main()
