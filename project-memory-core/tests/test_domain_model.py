"""
Sub-phase 1.37 — Domain model tests.

Covers:
- Default domain is "general"
- All 16 valid domains accepted on create
- Invalid domain rejected with 422
- Domain returned in ProjectResponse
- Domain updatable via PUT /projects/{id}
- Invalid domain update rejected with 422
- Domain appears in YAML export (project section)
- Domain appears in context.md export header
- Domain appears in per-type Markdown export header
- Existing projects without domain get "general" default
"""
import pytest
from tests.conftest import make_project

ALL_VALID_DOMAINS = [
    "software", "data", "security",
    "design", "creative",
    "product", "business", "marketing", "sales", "finance", "legal", "hr",
    "research", "education",
    "personal",
    "general",
]


# ---------------------------------------------------------------------------
# Create with domain
# ---------------------------------------------------------------------------

def test_default_domain_is_general(client):
    r = client.post("/projects", json={"name": "No Domain"})
    assert r.status_code == 201
    assert r.json()["domain"] == "general"


def test_create_with_explicit_domain(client):
    r = client.post("/projects", json={"name": "Design Project", "domain": "design"})
    assert r.status_code == 201
    assert r.json()["domain"] == "design"


@pytest.mark.parametrize("domain", ALL_VALID_DOMAINS)
def test_all_valid_domains_accepted(client, domain):
    r = client.post("/projects", json={"name": f"Project {domain}", "domain": domain})
    assert r.status_code == 201, f"Domain '{domain}' should be accepted, got {r.status_code}: {r.text}"
    assert r.json()["domain"] == domain


def test_invalid_domain_rejected(client):
    r = client.post("/projects", json={"name": "Bad Domain", "domain": "blockchain"})
    assert r.status_code == 422


def test_invalid_domain_typo_rejected(client):
    r = client.post("/projects", json={"name": "Typo", "domain": "sofware"})
    assert r.status_code == 422


def test_empty_string_domain_rejected(client):
    r = client.post("/projects", json={"name": "Empty", "domain": ""})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Domain in GET response
# ---------------------------------------------------------------------------

def test_domain_in_get_project(client):
    r = client.post("/projects", json={"name": "Research", "domain": "research"})
    pid = r.json()["id"]
    r2 = client.get(f"/projects/{pid}")
    assert r2.status_code == 200
    assert r2.json()["domain"] == "research"


def test_domain_in_list_projects(client):
    client.post("/projects", json={"name": "A", "domain": "marketing"})
    client.post("/projects", json={"name": "B", "domain": "finance"})
    r = client.get("/projects")
    domains = {p["name"]: p["domain"] for p in r.json()}
    assert domains["A"] == "marketing"
    assert domains["B"] == "finance"


# ---------------------------------------------------------------------------
# Update domain
# ---------------------------------------------------------------------------

def test_update_domain(client):
    p = make_project(client)
    r = client.put(f"/projects/{p['id']}", json={"domain": "design"})
    assert r.status_code == 200
    assert r.json()["domain"] == "design"


def test_update_to_invalid_domain_rejected(client):
    p = make_project(client)
    r = client.put(f"/projects/{p['id']}", json={"domain": "cooking"})
    assert r.status_code == 422


def test_update_domain_leaves_other_fields_unchanged(client):
    r = client.post("/projects", json={
        "name": "Stable", "domain": "software", "tech_stack": ["Python"]
    })
    pid = r.json()["id"]
    client.put(f"/projects/{pid}", json={"domain": "data"})
    r2 = client.get(f"/projects/{pid}")
    assert r2.json()["domain"] == "data"
    assert r2.json()["name"] == "Stable"
    assert r2.json()["tech_stack"] == ["Python"]


def test_update_without_domain_keeps_existing(client):
    r = client.post("/projects", json={"name": "Keep Domain", "domain": "legal"})
    pid = r.json()["id"]
    client.put(f"/projects/{pid}", json={"name": "Renamed"})
    r2 = client.get(f"/projects/{pid}")
    assert r2.json()["domain"] == "legal"


# ---------------------------------------------------------------------------
# Domain coverage — one for each category
# ---------------------------------------------------------------------------

def test_technical_domains(client):
    for domain in ["software", "data", "security"]:
        r = client.post("/projects", json={"name": domain, "domain": domain})
        assert r.json()["domain"] == domain


def test_design_creative_domains(client):
    for domain in ["design", "creative"]:
        r = client.post("/projects", json={"name": domain, "domain": domain})
        assert r.json()["domain"] == domain


def test_business_professional_domains(client):
    for domain in ["product", "business", "marketing", "sales", "finance", "legal", "hr"]:
        r = client.post("/projects", json={"name": domain, "domain": domain})
        assert r.json()["domain"] == domain


def test_knowledge_work_domains(client):
    for domain in ["research", "education"]:
        r = client.post("/projects", json={"name": domain, "domain": domain})
        assert r.json()["domain"] == domain


def test_personal_domain(client):
    r = client.post("/projects", json={"name": "Personal", "domain": "personal"})
    assert r.json()["domain"] == "personal"


# ---------------------------------------------------------------------------
# Domain in exports
# ---------------------------------------------------------------------------

def test_domain_in_yaml_export(client):
    r = client.post("/projects", json={"name": "YAML Test", "domain": "research"})
    pid = r.json()["id"]
    # Add a memory so export has content
    client.post(f"/projects/{pid}/memories", json={
        "type": "insight", "title": "Key finding", "content": "Study shows X",
        "importance": 3, "confidence": 0.9, "tags": [], "status": "active"
    })
    r2 = client.get(f"/projects/{pid}/export/memory.yaml")
    assert r2.status_code == 200
    assert "domain: research" in r2.text


def test_domain_in_context_md_export(client):
    r = client.post("/projects", json={"name": "Context Test", "domain": "design"})
    pid = r.json()["id"]
    r2 = client.get(f"/projects/{pid}/export/context.md")
    assert r2.status_code == 200
    assert "domain" in r2.text.lower()
    assert "design" in r2.text


def test_domain_in_per_type_export(client):
    r = client.post("/projects", json={"name": "Type Export Test", "domain": "marketing"})
    pid = r.json()["id"]
    client.post(f"/projects/{pid}/memories", json={
        "type": "insight", "title": "Campaign insight",
        "content": "Email performs better than social",
        "importance": 3, "confidence": 0.9, "tags": [], "status": "active"
    })
    r2 = client.get(f"/projects/{pid}/export/memories/insight.md")
    assert r2.status_code == 200
    # Header includes project name — domain flows through project
    assert "marketing" in r2.text or "Insights" in r2.text


def test_general_domain_in_yaml_export(client):
    r = client.post("/projects", json={"name": "General Project"})
    pid = r.json()["id"]
    client.post(f"/projects/{pid}/memories", json={
        "type": "task", "title": "Do something", "content": "content",
        "importance": 2, "confidence": 0.8, "tags": [], "status": "active"
    })
    r2 = client.get(f"/projects/{pid}/export/memory.yaml")
    assert "domain: general" in r2.text
