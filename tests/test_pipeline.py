from pathlib import Path
import spinegen


def main():
    data_dir = Path(__file__).parent.parent / "data" / "real"
    swc_files = list(data_dir.glob("*.swc"))

    print(f"Found {len(swc_files)} SWC files")

    print("\n=== Loading CableGraph models ===")
    graphs = []
    for swc_file in swc_files:
        print(f"Loading {swc_file.name}...")
        graph = spinegen.CableGraph.from_swc_file(swc_file)
        print(f"  Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
        graphs.append(graph)

    print("\n=== Analyzing morphology ===")
    analyzer = spinegen.SpineAnalyzer(graphs)
    stats = analyzer.analyze()

    print(f"Segment lengths: {len(stats['segment_lengths'])} samples")
    print(f"Curvatures: {len(stats['curvatures'])} samples")
    print(f"Branch counts: {len(stats['branch_counts'])} samples")
    print(f"Fusion distances: {len(stats['fusion_distances'])} samples")

    print("\n=== Creating prior ===")
    prior = spinegen.SpinePrior(stats)
    print(f"Fitted distributions: {list(prior.distributions.keys())}")

    print("\n=== Generating synthetic spine ===")
    generator = spinegen.SpineGenerator(prior)
    synthetic_graph = generator.generate(max_steps=100, max_nodes=50)

    print(f"Generated graph:")
    print(f"  Nodes: {synthetic_graph.number_of_nodes()}")
    print(f"  Edges: {synthetic_graph.number_of_edges()}")

    output_dir = Path(__file__).parent.parent / "data" / "synthetic"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "synthetic_spine.swc"
    print(f"\n=== Saving to {output_file.relative_to(Path.cwd())} ===")
    synthetic_graph.to_swc_file(output_file)
    print("Done!")


if __name__ == "__main__":
    main()
