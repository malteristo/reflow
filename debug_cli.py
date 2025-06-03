#!/usr/bin/env python3

from unittest.mock import patch
from typer.testing import CliRunner
from research_agent_backend.cli.cli import app

def debug_cli_command():
    with patch('research_agent_backend.core.augmentation_service.AugmentationService._calculate_quality_score') as mock_quality_score:
        from research_agent_backend.core.augmentation_service import QualityMetrics
        mock_quality_score.return_value = QualityMetrics(
            overall_score=0.85,  # High quality score to pass validation
            content_length_score=0.8,
            structure_score=0.9,
            credibility_score=0.8,
            freshness_score=0.9,
            relevance_score=0.8,
            language_score=0.9,
            uniqueness_score=0.8
        )
        
        runner = CliRunner()
        result = runner.invoke(app, [
            'kb', 'add-external-result',
            '--source', 'https://example.com/article',
            '--title', 'Machine Learning Advances in Computer Vision',
            '--content', 'This comprehensive research paper explores the latest advances in machine learning, specifically focusing on computer vision applications. The study examines various deep learning architectures including convolutional neural networks, transformer models, and their applications in image recognition, object detection, and semantic segmentation. Key findings demonstrate significant improvements in accuracy and computational efficiency through novel architectural innovations and training methodologies.',
            '--collection', 'research'
        ])
        
        print(f'EXIT CODE: {result.exit_code}')
        print(f'STDOUT: {result.stdout}')
        print(f'STDERR: {result.stderr if result.stderr else "None"}')
        if result.exception:
            print(f'EXCEPTION: {str(result.exception)}')
            import traceback
            print('TRACEBACK:')
            traceback.print_exception(type(result.exception), result.exception, result.exception.__traceback__)

if __name__ == "__main__":
    debug_cli_command() 