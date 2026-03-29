"""datadiffusion — entry point."""

from datadiffusion.cli import parse_args
from datadiffusion.tracking.logger import setup_logging
from datadiffusion.pipeline.self_improvement import SelfImprovementLoop

if __name__ == "__main__":
    config = parse_args()
    setup_logging(verbose=config.verbose)

    loop = SelfImprovementLoop(config)
    result = loop.run()

    print(f"\nBest composite score: {result.quality.composite_score:.4f}")
