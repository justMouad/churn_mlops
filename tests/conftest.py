import pytest


@pytest.fixture(scope="session")
def artifacts(tmp_path_factory):
    """Entraîne un jeu d'artefacts léger une seule fois pour toute la session de tests."""
    from churnops.pipeline import run_training
    d = tmp_path_factory.mktemp("artifacts")
    run_training(d, quick=True, verbose=False)
    return d


@pytest.fixture(scope="session")
def client(artifacts):
    from fastapi.testclient import TestClient

    # le lifespan recharge ModelState() ; on le pointe vers les artefacts de test
    import app.main as m
    from app.main import app
    from app.state import ModelState
    m.ModelState = lambda: ModelState(artifacts)
    with TestClient(app) as c:
        yield c
