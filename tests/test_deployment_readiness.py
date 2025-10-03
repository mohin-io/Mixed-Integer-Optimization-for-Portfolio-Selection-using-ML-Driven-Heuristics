"""
Deployment Readiness Tests

Verifies that the application is ready for production deployment.
Tests file structure, dependencies, configuration, and critical functionality.
"""

import pytest
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestFileStructure:
    """Test that all required files and directories exist."""

    def test_project_root_exists(self):
        """Test that project root directory exists."""
        root = Path(__file__).parent.parent
        assert root.exists(), "Project root directory not found"

    def test_src_directory_exists(self):
        """Test that src directory exists."""
        src_dir = Path(__file__).parent.parent / 'src'
        assert src_dir.exists(), "src/ directory not found"

    def test_dashboard_file_exists(self):
        """Test that main dashboard file exists."""
        dashboard = Path(__file__).parent.parent / 'src' / 'visualization' / 'dashboard.py'
        assert dashboard.exists(), "dashboard.py not found"
        assert dashboard.stat().st_size > 0, "dashboard.py is empty"

    def test_requirements_file_exists(self):
        """Test that requirements.txt exists."""
        requirements = Path(__file__).parent.parent / 'requirements.txt'
        assert requirements.exists(), "requirements.txt not found"
        assert requirements.stat().st_size > 0, "requirements.txt is empty"

    def test_readme_exists(self):
        """Test that README exists."""
        readme = Path(__file__).parent.parent / 'README.md'
        assert readme.exists(), "README.md not found"

    def test_streamlit_config_exists(self):
        """Test that Streamlit config exists."""
        config = Path(__file__).parent.parent / '.streamlit' / 'config.toml'
        assert config.exists(), ".streamlit/config.toml not found"

    def test_deployment_docs_exist(self):
        """Test that deployment documentation exists."""
        deployment_md = Path(__file__).parent.parent / 'docs' / 'DEPLOYMENT.md'
        assert deployment_md.exists(), "docs/DEPLOYMENT.md not found"

        deployment_steps = Path(__file__).parent.parent / 'docs' / 'DEPLOYMENT_STEPS.md'
        assert deployment_steps.exists(), "docs/DEPLOYMENT_STEPS.md not found"

    def test_user_guide_exists(self):
        """Test that user guide exists."""
        user_guide = Path(__file__).parent.parent / 'docs' / 'USER_GUIDE.md'
        assert user_guide.exists(), "docs/USER_GUIDE.md not found"

    def test_test_files_exist(self):
        """Test that test files exist."""
        tests_dir = Path(__file__).parent
        assert (tests_dir / 'test_dashboard.py').exists(), "test_dashboard.py not found"
        assert (tests_dir / 'test_integration_dashboard.py').exists(), "test_integration_dashboard.py not found"


class TestDependencies:
    """Test that all required dependencies are installed."""

    def test_streamlit_installed(self):
        """Test that Streamlit is installed."""
        try:
            import streamlit
            assert hasattr(streamlit, '__version__')
        except ImportError:
            pytest.fail("Streamlit not installed")

    def test_numpy_installed(self):
        """Test that NumPy is installed."""
        try:
            import numpy as np
            assert hasattr(np, '__version__')
        except ImportError:
            pytest.fail("NumPy not installed")

    def test_pandas_installed(self):
        """Test that Pandas is installed."""
        try:
            import pandas as pd
            assert hasattr(pd, '__version__')
        except ImportError:
            pytest.fail("Pandas not installed")

    def test_matplotlib_installed(self):
        """Test that Matplotlib is installed."""
        try:
            import matplotlib
            assert hasattr(matplotlib, '__version__')
        except ImportError:
            pytest.fail("Matplotlib not installed")

    def test_seaborn_installed(self):
        """Test that Seaborn is installed."""
        try:
            import seaborn
            assert hasattr(seaborn, '__version__')
        except ImportError:
            pytest.fail("Seaborn not installed")

    def test_pytest_installed(self):
        """Test that pytest is installed."""
        try:
            import pytest
            assert hasattr(pytest, '__version__')
        except ImportError:
            pytest.fail("pytest not installed")


class TestConfiguration:
    """Test configuration files."""

    def test_requirements_content(self):
        """Test that requirements.txt has necessary packages."""
        requirements_path = Path(__file__).parent.parent / 'requirements.txt'
        content = requirements_path.read_text()

        required_packages = [
            'streamlit',
            'numpy',
            'pandas',
            'matplotlib',
            'seaborn'
        ]

        for package in required_packages:
            assert package in content.lower(), f"{package} not in requirements.txt"

    def test_streamlit_config_valid(self):
        """Test that Streamlit config is valid."""
        config_path = Path(__file__).parent.parent / '.streamlit' / 'config.toml'
        content = config_path.read_text()

        # Check for essential configuration
        assert '[theme]' in content or '[server]' in content, \
            "Streamlit config missing theme or server section"

    def test_no_secrets_in_code(self):
        """Test that no obvious secrets are in the codebase."""
        dashboard_path = Path(__file__).parent.parent / 'src' / 'visualization' / 'dashboard.py'
        content = dashboard_path.read_text(encoding='utf-8').lower()

        # Check for common secret patterns
        forbidden_patterns = [
            'api_key =',
            'password =',
            'secret =',
            'token =',
            'aws_access_key'
        ]

        for pattern in forbidden_patterns:
            assert pattern not in content, f"Potential secret found: {pattern}"


class TestDashboardFunctionality:
    """Test critical dashboard functionality."""

    def test_dashboard_imports_successfully(self):
        """Test that dashboard module imports without errors."""
        try:
            from src.visualization import dashboard
            assert True
        except Exception as e:
            pytest.fail(f"Dashboard import failed: {e}")

    def test_main_function_exists(self):
        """Test that main function exists."""
        from src.visualization.dashboard import main
        assert callable(main), "main() function not callable"

    def test_generate_data_function_exists(self):
        """Test that data generation function exists."""
        from src.visualization.dashboard import generate_synthetic_data
        assert callable(generate_synthetic_data), "generate_synthetic_data() not callable"

    def test_optimize_function_exists(self):
        """Test that optimize function exists."""
        from src.visualization.dashboard import optimize_portfolio
        assert callable(optimize_portfolio), "optimize_portfolio() not callable"

    def test_evaluate_function_exists(self):
        """Test that evaluate function exists."""
        from src.visualization.dashboard import evaluate_portfolio
        assert callable(evaluate_portfolio), "evaluate_portfolio() not callable"

    def test_basic_workflow_executes(self):
        """Test that basic workflow executes without errors."""
        from src.visualization.dashboard import (
            generate_synthetic_data,
            optimize_portfolio,
            evaluate_portfolio
        )

        # Generate data
        prices, returns = generate_synthetic_data(5, 100, 42)

        # Optimize
        weights, annual_returns, cov_matrix = optimize_portfolio(returns, 'Equal Weight')

        # Evaluate
        metrics = evaluate_portfolio(weights, annual_returns, cov_matrix)

        # Verify results
        assert metrics is not None
        assert 'return' in metrics
        assert 'volatility' in metrics
        assert 'sharpe' in metrics


class TestDocumentation:
    """Test that documentation is complete."""

    def test_readme_has_content(self):
        """Test that README has substantial content."""
        readme = Path(__file__).parent.parent / 'README.md'
        content = readme.read_text(encoding='utf-8')

        assert len(content) > 500, "README too short"
        assert '# ' in content, "README missing headers"

    def test_readme_has_quickstart(self):
        """Test that README has quickstart section."""
        readme = Path(__file__).parent.parent / 'README.md'
        content = readme.read_text(encoding='utf-8').lower()

        assert 'quickstart' in content or 'getting started' in content or 'installation' in content, \
            "README missing quickstart/installation section"

    def test_deployment_docs_complete(self):
        """Test that deployment docs are complete."""
        deployment = Path(__file__).parent.parent / 'docs' / 'DEPLOYMENT.md'
        content = deployment.read_text(encoding='utf-8').lower()

        deployment_platforms = ['streamlit', 'heroku', 'docker', 'aws']
        for platform in deployment_platforms:
            assert platform in content, f"Deployment docs missing {platform} instructions"

    def test_user_guide_complete(self):
        """Test that user guide is complete."""
        user_guide = Path(__file__).parent.parent / 'docs' / 'USER_GUIDE.md'
        content = user_guide.read_text(encoding='utf-8').lower()

        required_sections = ['getting started', 'strategy', 'visualization']
        for section in required_sections:
            assert section in content, f"User guide missing {section} section"


class TestGitRepository:
    """Test Git repository status."""

    def test_git_repository_exists(self):
        """Test that .git directory exists."""
        git_dir = Path(__file__).parent.parent / '.git'
        assert git_dir.exists(), "Not a Git repository"

    def test_gitignore_exists(self):
        """Test that .gitignore exists."""
        gitignore = Path(__file__).parent.parent / '.gitignore'
        assert gitignore.exists(), ".gitignore not found"

    def test_gitignore_has_python_patterns(self):
        """Test that .gitignore includes Python patterns."""
        gitignore = Path(__file__).parent.parent / '.gitignore'
        content = gitignore.read_text(encoding='utf-8')

        # Check for essential Python patterns (*.py[cod] covers *.pyc)
        python_patterns = ['__pycache__', '.env', 'venv']
        for pattern in python_patterns:
            assert pattern in content, f".gitignore missing {pattern}"

        # Check that either *.pyc or *.py[cod] is present
        assert '*.pyc' in content or '*.py[cod]' in content, \
            ".gitignore missing Python bytecode pattern"


class TestDeploymentReadiness:
    """High-level deployment readiness tests."""

    def test_all_strategies_work(self):
        """Test that all optimization strategies work."""
        from src.visualization.dashboard import generate_synthetic_data, optimize_portfolio

        prices, returns = generate_synthetic_data(10, 252, 42)

        strategies = ['Equal Weight', 'Max Sharpe', 'Min Variance', 'Concentrated']

        for strategy in strategies:
            if strategy == 'Concentrated':
                weights, _, _ = optimize_portfolio(returns, strategy, max_assets=5)
            else:
                weights, _, _ = optimize_portfolio(returns, strategy)

            assert weights is not None, f"{strategy} failed"
            assert abs(weights.sum() - 1.0) < 1e-6, f"{strategy} weights don't sum to 1"

    def test_no_warnings_on_import(self):
        """Test that importing dashboard doesn't raise warnings."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from src.visualization import dashboard

            # Filter out Streamlit-specific warnings which are expected
            critical_warnings = [warning for warning in w
                               if 'streamlit' not in str(warning.message).lower()]

            assert len(critical_warnings) == 0, \
                f"Unexpected warnings on import: {[str(w.message) for w in critical_warnings]}"

    def test_performance_acceptable(self):
        """Test that optimization performance is acceptable."""
        import time
        from src.visualization.dashboard import generate_synthetic_data, optimize_portfolio

        prices, returns = generate_synthetic_data(10, 252, 42)

        start = time.time()
        weights, _, _ = optimize_portfolio(returns, 'Max Sharpe')
        elapsed = time.time() - start

        assert elapsed < 30, f"Optimization too slow: {elapsed:.2f}s (max 30s)"

    def test_memory_usage_reasonable(self):
        """Test that memory usage is reasonable."""
        import sys
        from src.visualization.dashboard import generate_synthetic_data, optimize_portfolio

        prices, returns = generate_synthetic_data(20, 2000, 42)
        weights, annual_returns, cov_matrix = optimize_portfolio(returns, 'Equal Weight')

        # Check sizes
        prices_size = sys.getsizeof(prices) / 1024 / 1024  # MB
        returns_size = sys.getsizeof(returns) / 1024 / 1024  # MB
        weights_size = sys.getsizeof(weights) / 1024 / 1024  # MB

        total_size = prices_size + returns_size + weights_size

        # Should be less than 10MB for reasonable data sizes
        assert total_size < 10, f"Memory usage too high: {total_size:.2f}MB"


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_handles_single_asset(self):
        """Test that single asset is handled correctly."""
        import pandas as pd
        import numpy as np
        from src.visualization.dashboard import optimize_portfolio

        returns = pd.DataFrame({'ASSET_1': np.random.randn(100) * 0.01})
        weights, _, _ = optimize_portfolio(returns, 'Equal Weight')

        assert weights['ASSET_1'] == pytest.approx(1.0), "Single asset should get 100% weight"

    def test_handles_zero_volatility(self):
        """Test handling of zero volatility edge case."""
        import pandas as pd
        import numpy as np
        from src.visualization.dashboard import evaluate_portfolio

        weights = pd.Series([1.0], index=['A1'])
        annual_returns = pd.Series([0.1], index=['A1'])
        cov_matrix = pd.DataFrame([[0.0]], index=['A1'], columns=['A1'])

        metrics = evaluate_portfolio(weights, annual_returns, cov_matrix)

        # Should handle gracefully without errors
        assert metrics['sharpe'] == 0, "Sharpe should be 0 when volatility is 0"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
