"""
Tests smoke pour le pipeline GitLab CI
"""
def test_smoke():
    """Test basique pour vérifier que pytest fonctionne"""
    assert 1 + 1 == 2

def test_imports():
    """Test que les imports principaux fonctionnent"""
    try:
        import fastapi
        import sklearn
        import pandas
        import mlflow
        import optuna
        assert True
    except ImportError as e:
        assert False, f"Import failed: {e}"
