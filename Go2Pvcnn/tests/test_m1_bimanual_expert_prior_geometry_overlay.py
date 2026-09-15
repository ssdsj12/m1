"""Fail-closed verification of independently pinned geometry overlays."""
import hashlib
import json

import pytest

from go2_pvcnn.control.m1_bimanual_coordination.expert_fingertip_prior.geometry_overlay import verify_geometry_overlay


def sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def valid_overlay(tmp_path):
    (tmp_path / "objects").mkdir()
    mesh = tmp_path / "objects/mesh.obj"
    mesh.write_bytes(b"v 0 0 0\n")
    urdf = tmp_path / "objects/object.urdf"
    urdf.write_text('<robot name="object"><link name="base"><visual><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="mesh.obj" scale="1 1 1"/></geometry></visual><collision><geometry><mesh filename="mesh.obj"/></geometry></collision></link></robot>')
    reference = "ObjURDF/align_ds/object/model.urdf"
    data = {"schema_version": 1, "upstream": {"repository": "kelvin34501/OakInk-v2", "revision": "21705616140d726607027e70d58b7837f442ffd8", "archive_sha256": "40bb71fb59e1288e5673f32c5bd8fdb501bef15e8b666005b4a6349549983cd2"}, "recipe": {"tool": "fixture", "version": "1", "arguments": {"seed": 1}}, "entries": [{"reference": reference, "raw_member": "object_raw/align_ds/object.obj", "raw_sha256": "a" * 64, "urdf": "objects/object.urdf", "urdf_sha256": sha(urdf.read_bytes()), "meshes": [{"path": "objects/mesh.obj", "sha256": sha(mesh.read_bytes())}], "checks": {"fixture": True}}]}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(data))
    return tmp_path, manifest, reference


def verify(fixture):
    root, manifest, _ = fixture
    return verify_geometry_overlay(root, manifest, sha(manifest.read_bytes()))


def edit_manifest(fixture, edit):
    _, manifest, _ = fixture
    data = json.loads(manifest.read_bytes())
    edit(data)
    manifest.write_text(json.dumps(data))


def test_success_and_exact_external_identity(valid_overlay):
    root, manifest, reference = valid_overlay
    resolver = verify(valid_overlay)
    assert resolver.manifest_sha256 == sha(manifest.read_bytes())
    assert resolver.resolve(reference, source="oakinkv2") == root / "objects/object.urdf"
    with pytest.raises(AttributeError):
        resolver.manifest_sha256 = "0" * 64


def test_wrong_external_pin_rejected(valid_overlay):
    root, manifest, _ = valid_overlay
    with pytest.raises(ValueError):
        verify_geometry_overlay(root, manifest, "0" * 64)


@pytest.mark.parametrize("recipe", [
    {"unrelated": True},
    {"tool": None, "version": "1", "arguments": {"seed": 1}},
    {"tool": "", "version": "1", "arguments": {"seed": 1}},
    {"tool": " fixture ", "version": "1", "arguments": {"seed": 1}},
    {"tool": "fixture", "version": None, "arguments": {"seed": 1}},
    {"tool": "fixture", "version": " ", "arguments": {"seed": 1}},
    {"tool": "fixture", "version": "1"},
    {"tool": "fixture", "version": "1", "arguments": []},
    {"tool": "fixture", "version": "1", "arguments": {}},
    {"tool": "fixture", "version": "1", "arguments": {"": 1}},
])
def test_recipe_requires_tool_version_and_arguments(valid_overlay, recipe):
    edit_manifest(valid_overlay, lambda data: data.__setitem__("recipe", recipe))
    with pytest.raises(ValueError):
        verify(valid_overlay)


@pytest.mark.parametrize("field", ["reference", "raw_member", "urdf"])
@pytest.mark.parametrize("path", ["/outside", "../outside", "a/../b", "a//b", "./a", "a/", "", "a\\b"])
def test_unsafe_paths_rejected_without_writes(valid_overlay, field, path):
    edit_manifest(valid_overlay, lambda data: data["entries"][0].__setitem__(field, path))
    before = {p: p.read_bytes() for p in valid_overlay[0].rglob("*") if p.is_file()}
    with pytest.raises(ValueError):
        verify(valid_overlay)
    assert before == {p: p.read_bytes() for p in before}


@pytest.mark.parametrize("edit", [
    lambda d: d.__setitem__("schema_version", True),
    lambda d: d.__setitem__("schema_version", 2),
    lambda d: d.__setitem__("extra", 1),
    lambda d: d.__setitem__("recipe", {}),
    lambda d: d["entries"][0].__setitem__("checks", {}),
    lambda d: d["entries"][0].__setitem__("raw_sha256", "A" * 64),
    lambda d: d["entries"][0].__setitem__("urdf_sha256", "bad"),
    lambda d: d["entries"][0]["meshes"][0].__setitem__("sha256", "0" * 64),
    lambda d: d["entries"].append(d["entries"][0].copy()),
    lambda d: d["entries"][0]["meshes"].append(d["entries"][0]["meshes"][0].copy()),
    lambda d: d["entries"][0]["meshes"][0].__setitem__("path", "../mesh.obj"),
    lambda d: d["entries"][0].__setitem__("meshes", []),
])
def test_bad_schema_rejected(valid_overlay, edit):
    edit_manifest(valid_overlay, edit)
    with pytest.raises(ValueError):
        verify(valid_overlay)


@pytest.mark.parametrize("field", ["repository", "revision", "archive_sha256"])
def test_wrong_upstream_rejected(valid_overlay, field):
    edit_manifest(valid_overlay, lambda d: d["upstream"].__setitem__(field, "wrong"))
    with pytest.raises(ValueError):
        verify(valid_overlay)


@pytest.mark.parametrize("payload", ['{"schema_version":1,"schema_version":1}', '{"recipe":{"a":1,"a":2}}', '{"a":NaN}', '{"a":Infinity}', '{"a":1e999}', '[]', '{'])
def test_strict_json(valid_overlay, payload):
    valid_overlay[1].write_text(payload)
    with pytest.raises(ValueError):
        verify(valid_overlay)


@pytest.mark.parametrize("filename", ["object.urdf", "mesh.obj"])
def test_missing_or_drifting_files(valid_overlay, filename):
    path = valid_overlay[0] / "objects" / filename
    path.write_bytes(path.read_bytes() + b"drift")
    with pytest.raises(ValueError):
        verify(valid_overlay)
    path.unlink()
    with pytest.raises(ValueError):
        verify(valid_overlay)


@pytest.mark.parametrize("filename", ["object.urdf", "mesh.obj"])
def test_resolve_rechecks_bytes(valid_overlay, filename):
    resolver = verify(valid_overlay)
    (valid_overlay[0] / "objects" / filename).write_bytes(b"drift")
    with pytest.raises(ValueError):
        resolver.resolve(valid_overlay[2], source="oakinkv2")


@pytest.mark.parametrize("source", ["favor", "OakInkV2", "", None])
def test_non_oakink_source(valid_overlay, source):
    with pytest.raises(ValueError):
        verify(valid_overlay).resolve(valid_overlay[2], source=source)


def test_unknown_reference(valid_overlay):
    with pytest.raises(ValueError):
        verify(valid_overlay).resolve("unknown.urdf", source="oakinkv2")


@pytest.mark.parametrize("old,new", [
    ('scale="1 1 1"', 'scale="1 2 1"'),
    ('xyz="0 0 0"', 'xyz="0 0 1"'),
    ('rpy="0 0 0"', 'rpy="0 1 0"'),
    ('scale="1 1 1"', 'scale="nan 1 1"'),
    ('filename="mesh.obj"', 'filename="../objects/mesh.obj"'),
    ('filename="mesh.obj"', 'filename="undeclared.obj"'),
    ('<collision>', '<visual>'),
    ('</collision>', '</visual>'),
])
def test_invalid_urdf_declarations(valid_overlay, old, new):
    path = valid_overlay[0] / "objects/object.urdf"
    path.write_text(path.read_text().replace(old, new))
    edit_manifest(valid_overlay, lambda d: d["entries"][0].__setitem__("urdf_sha256", sha(path.read_bytes())))
    with pytest.raises(ValueError):
        verify(valid_overlay)


@pytest.mark.parametrize("target", ["manifest", "parent", "mesh", "root"])
def test_contained_symlinks_rejected(valid_overlay, target):
    root, manifest, _ = valid_overlay
    if target == "root":
        link = root.parent / (root.name + "-link")
        link.symlink_to(root, target_is_directory=True)
        fixture = (link, link / "manifest.json", valid_overlay[2])
    else:
        path = {"manifest": manifest, "parent": root / "objects", "mesh": root / "objects/mesh.obj"}[target]
        real = path.with_name(path.name + "-real")
        path.rename(real)
        path.symlink_to(real, target_is_directory=target == "parent")
        fixture = valid_overlay
    with pytest.raises(ValueError):
        verify(fixture)


def test_resolve_rejects_new_symlink_parent(valid_overlay):
    resolver = verify(valid_overlay)
    directory = valid_overlay[0] / "objects"
    real = valid_overlay[0] / "moved"
    directory.rename(real)
    directory.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError):
        resolver.resolve(valid_overlay[2], source="oakinkv2")


def test_utf16_entity_declaration_rejected(valid_overlay):
    path = valid_overlay[0] / "objects/object.urdf"
    xml = path.read_text().replace('filename="mesh.obj"', 'filename="&asset;"')
    path.write_bytes(('<!DOCTYPE robot [<!ENTITY asset "mesh.obj">]>' + xml).encode("utf-16"))
    edit_manifest(valid_overlay, lambda d: d["entries"][0].__setitem__("urdf_sha256", sha(path.read_bytes())))
    with pytest.raises(ValueError):
        verify(valid_overlay)


def test_well_formed_urdf_without_collision_rejected(valid_overlay):
    path = valid_overlay[0] / "objects/object.urdf"
    path.write_text(path.read_text().replace("<collision>", "<visual>").replace("</collision>", "</visual>"))
    edit_manifest(valid_overlay, lambda d: d["entries"][0].__setitem__("urdf_sha256", sha(path.read_bytes())))
    with pytest.raises(ValueError):
        verify(valid_overlay)


def test_manifest_outside_root_rejected(valid_overlay, tmp_path):
    root, manifest, _ = valid_overlay
    outside = root.parent / (root.name + "-manifest.json")
    outside.write_bytes(manifest.read_bytes())
    with pytest.raises(ValueError):
        verify_geometry_overlay(root, outside, sha(outside.read_bytes()))


@pytest.mark.parametrize("pin", [None, "A" * 64, "short"])
def test_malformed_external_pin(valid_overlay, pin):
    with pytest.raises(ValueError):
        verify_geometry_overlay(valid_overlay[0], valid_overlay[1], pin)


def test_private_records_are_read_only(valid_overlay):
    resolver = verify(valid_overlay)
    with pytest.raises(TypeError):
        resolver._entries[valid_overlay[2]] = None
    with pytest.raises(AttributeError):
        resolver._entries[valid_overlay[2]].urdf = "other.urdf"
