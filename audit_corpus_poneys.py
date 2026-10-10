import os
cd#!/usr/bin/env python3
"""Audit en lecture seule. Python >= 3.9; pip install Pillow ImageHash.

Les groupes de similarité sont des candidats automatiques, pas des séances
certifiées. Aucune image n'est déplacée, supprimée ou réécrite.
"""
import argparse
import csv
import hashlib
import json
import statistics
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    from PIL import Image, ImageOps
    import imagehash
except ImportError:
    raise SystemExit("Installez les dépendances : python -m pip install Pillow ImageHash")

BASE = Path(os.environ.get("EQUINE_DATA_ROOT", "dataset"))
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}


def normalized_name(name):
    return unicodedata.normalize("NFKC", name).strip().casefold()


def sha_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class BKTree:
    """Index des pHash pour rechercher des voisins sans comparer toutes les paires."""
    def __init__(self):
        self.root = None

    def insert(self, value, idx):
        if self.root is None:
            self.root = [value, [idx], {}]
            return
        node = self.root
        while True:
            d = bin(value ^ node[0]).count("1")
            if d == 0:
                node[1].append(idx)
                return
            if d not in node[2]:
                node[2][d] = [value, [idx], {}]
                return
            node = node[2][d]

    def neighbors(self, value, threshold):
        stack = [self.root] if self.root else []
        while stack:
            node = stack.pop()
            d = bin(value ^ node[0]).count("1")
            if d <= threshold:
                for idx in node[1]:
                    yield idx, d
            stack.extend(child for edge, child in node[2].items()
                         if d - threshold <= edge <= d + threshold)


def write_csv(path, rows, fields):
    # utf-8-sig et point-virgule pour une ouverture simple dans Excel français.
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter=";", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def audit(roots, out, threshold=4):
    for category, root in roots:
        if not root.is_dir():
            raise ValueError(f"Répertoire absent : {root}")
    # Les rapports ne doivent pas devenir des entrées lors d'un nouvel audit.
    if any(out.resolve() == root.resolve() or root.resolve() in out.resolve().parents
           for _, root in roots):
        raise ValueError("Le dossier de sortie doit être hors des deux répertoires d'images.")
    out.mkdir(parents=True, exist_ok=True)
    records, folders, ignored, root_images = [], [], [], []
    for category, root in roots:
        for f in sorted(root.iterdir()):
            if f.is_file() and f.suffix.lower() in EXTENSIONS:
                root_images.append(str(f))
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            identity = f"{category}/{folder.name}"
            folders.append({"identity": identity, "poney": folder.name,
                            "categorie": category, "dossier": str(folder)})
            for path in sorted(folder.rglob("*")):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in EXTENSIONS:
                    ignored.append(str(path))
                    continue
                row = {"index": len(records), "identity": identity, "poney": folder.name,
                       "categorie": category, "path": str(path), "relative_path": str(path.relative_to(root)),
                       "status": "ok", "error": "", "sha256": "", "pixel_sha256": "",
                       "phash": "", "width": "", "height": "", "exif_date": "",
                       "frames": "", "candidate_group": ""}
                try:
                    row["sha256"] = sha_file(path)
                    with Image.open(path) as im:
                        row["frames"] = getattr(im, "n_frames", 1)
                        exif = im.getexif()
                        row["exif_date"] = str(exif.get(36867) or exif.get(306) or "")
                        rgb = ImageOps.exif_transpose(im).convert("RGB")
                        rgb.load()
                        row["width"], row["height"] = rgb.size
                        h = hashlib.sha256()
                        h.update(f"{rgb.width}x{rgb.height}:RGB:".encode("ascii"))
                        h.update(rgb.tobytes())
                        row["pixel_sha256"] = h.hexdigest()
                        row["phash"] = str(imagehash.phash(rgb, hash_size=8))
                except Exception as e:
                    row["status"], row["error"] = "illisible", f"{type(e).__name__}: {e}"
                records.append(row)
                if len(records) % 100 == 0:
                    print(f"{len(records)} fichiers examinés...", flush=True)

    def duplicates(key):
        groups = defaultdict(list)
        for r in records:
            if r[key]:
                groups[r[key]].append(r["index"])
        return [{"hash": k, "indices": ids,
                 "identites": sorted({records[i]["identity"] for i in ids}),
                 "inter_identites": len({records[i]["identity"] for i in ids}) > 1}
                for k, ids in groups.items() if len(ids) > 1]

    parent = list(range(len(records)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    # Chaque paire détectée est écrite; les groupes sont formés uniquement
    # à l'intérieur d'une identité. Les conflits entre identités sont signalés.
    tree = BKTree()
    pair_count, cross_count = 0, 0
    with (out / "similarites_candidates.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, delimiter=";", fieldnames=[
            "index_a", "index_b", "path_a", "path_b", "distance_phash", "inter_identites"])
        writer.writeheader()
        for r in records:
            if r["status"] != "ok":
                continue
            value = int(r["phash"], 16)
            for idx, distance in tree.neighbors(value, threshold):
                previous = records[idx]
                # Un même pixel_sha est un doublon certain; un pHash proche
                # est seulement un candidat. Aucun filtrage de ratio ne masque
                # ici les possibles copies recadrées, mais pHash peut les rater.
                cross = previous["identity"] != r["identity"]
                writer.writerow({"index_a": idx, "index_b": r["index"],
                                 "path_a": previous["path"], "path_b": r["path"],
                                 "distance_phash": distance, "inter_identites": cross})
                pair_count += 1
                cross_count += int(cross)
                if not cross:
                    union(idx, r["index"])
            tree.insert(value, r["index"])
    # Les doublons certains sont groupés même si l'image est très uniforme.
    pixels = duplicates("pixel_sha256")
    for g in pixels:
        by_identity = defaultdict(list)
        for idx in g["indices"]:
            by_identity[records[idx]["identity"]].append(idx)
        for ids in by_identity.values():
            for idx in ids[1:]:
                union(ids[0], idx)
    grouped = defaultdict(list)
    for r in records:
        if r["status"] == "ok":
            grouped[find(r["index"])].append(r["index"])
    candidate_groups = []
    for number, ids in enumerate(sorted(grouped.values(), key=lambda x: min(x)), 1):
        gid = f"G{number:06d}"
        for idx in ids:
            records[idx]["candidate_group"] = gid
        if len(ids) > 1:
            candidate_groups.append({"group": gid, "identity": records[ids[0]]["identity"], "indices": ids})

    by_id = defaultdict(list)
    for r in records:
        by_id[r["identity"]].append(r)
    pony_rows, anomalies = [], []
    names = defaultdict(list)
    for folder in folders:
        names[normalized_name(folder["poney"])].append(folder["identity"])
        rows = by_id[folder["identity"]]
        valid = [r for r in rows if r["status"] == "ok"]
        n = len(valid)
        row = dict(folder, fichiers_images=len(rows), images_lisibles=n,
                   images_illisibles=len(rows)-n,
                   contenus_pixels_uniques=len({r["pixel_sha256"] for r in valid}),
                   groupes_candidates=len({r["candidate_group"] for r in valid}),
                   photos_avec_date_exif=sum(bool(r["exif_date"]) for r in valid))
        pony_rows.append(row)
        if (folder["categorie"] == "au_moins_10_photos" and n < 10) or (
                folder["categorie"] == "moins_de_10_photos" and n >= 10):
            anomalies.append({"identity": folder["identity"], "raison": "catégorie incompatible avec le nombre de photos lisibles", "nombre": n})
        if n == 0:
            anomalies.append({"identity": folder["identity"], "raison": "aucune image lisible"})
    colliding_names = {k: v for k, v in names.items() if len(v) > 1}
    counts = [r["images_lisibles"] for r in pony_rows]
    byte_duplicates = duplicates("sha256")
    report = {
        "version": "1.0", "date_utc": datetime.now(timezone.utc).isoformat(),
        "racines": {k: str(v) for k, v in roots},
        "parametres": {"phash_bits": 64, "distance_max": threshold},
        "resume": {"dossiers_poneys": len(folders), "images_trouvees": len(records),
                   "images_lisibles": sum(r["status"] == "ok" for r in records),
                   "images_illisibles": sum(r["status"] != "ok" for r in records),
                   "contenus_pixels_uniques": len({r["pixel_sha256"] for r in records if r["status"] == "ok"}),
                   "photos_min": min(counts) if counts else 0,
                   "photos_mediane": statistics.median(counts) if counts else 0,
                   "photos_max": max(counts) if counts else 0,
                   "photos_moyenne": statistics.mean(counts) if counts else 0,
                   "dossiers_par_categorie": {k: sum(f["categorie"] == k for f in folders) for k, _ in roots},
                   "groupes_doublons_fichiers": len(byte_duplicates),
                   "groupes_doublons_pixels": len(pixels),
                   "paires_similaires_candidates": pair_count,
                   "paires_similaires_inter_identites": cross_count},
        "poneys": pony_rows, "doublons_fichiers": byte_duplicates,
        "doublons_pixels": pixels, "groupes_similaires_candidates": candidate_groups,
        "noms_presents_dans_plusieurs_dossiers": colliding_names,
        "anomalies": anomalies, "images_hors_dossiers_poneys": root_images,
        "fichiers_non_images_ignores": ignored,
        "images_illisibles": [r for r in records if r["status"] != "ok"],
        "limites": [
            "Les groupes pHash sont des candidats de proximité visuelle, pas des séances photographiques certifiées.",
            "Des photos distinctes peuvent être regroupées; des rafales, recadrages ou photos d'une même séance peuvent être manqués.",
            "Un groupe connexe peut contenir des images dont la distance mutuelle dépasse le seuil (chaînage).",
            "Les dates EXIF ne sont pas validées; les dates de modification des fichiers ne sont pas utilisées comme dates de prise de vue.",
            "Une image avec plusieurs poneys conserve l'identité du dossier, sans vérification automatique du sujet.",
            "Le nombre de groupes ne constitue pas une mesure du nombre de séances indépendantes.",
            "Les noms identiques dans plusieurs dossiers nécessitent une résolution avant les splits.",
            "Pour les GIF/TIFF multipages, seule la première image est auditée; le nombre de frames est indiqué."
        ]
    }
    (out / "bilan_corpus.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    image_fields = list(records[0]) if records else ["index", "identity", "path", "status"]
    write_csv(out / "inventaire_images.csv", records, image_fields)
    pony_fields = list(pony_rows[0]) if pony_rows else ["identity", "poney", "categorie", "images_lisibles"]
    write_csv(out / "inventaire_poneys.csv", pony_rows, pony_fields)
    summary = report["resume"]
    lines = ["AUDIT DU CORPUS DE PONEYS", "", *[f"{k} : {v}" for k, v in summary.items()],
             "", f"Noms présents dans plusieurs dossiers : {len(colliding_names)}",
             f"Anomalies de dossiers : {len(anomalies)}", "", "LIMITES", *report["limites"],
             "", "Aucune image n'a été modifiée. Aucun split d'évaluation n'a été créé.",
             "Transmettre bilan_corpus.json et inventaire_images.csv pour préparer les expériences."]
    (out / "resume_audit.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:len(summary)+2]))
    print(f"\nRapports créés dans : {out.resolve()}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=BASE, help="Dossier contenant les deux catégories")
    parser.add_argument("--output", type=Path, help="Dossier de rapports (défaut : BASE/audit_corpus)")
    parser.add_argument("--phash-distance", type=int, default=4, help="Seuil conservateur de candidats, entre 0 et 16")
    args = parser.parse_args()
    if not 0 <= args.phash_distance <= 16:
        parser.error("--phash-distance doit être compris entre 0 et 16")
    try:
        audit([(k, args.base / k) for k in ("au_moins_10_photos", "moins_de_10_photos")],
              args.output or args.base / "audit_corpus", args.phash_distance)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Erreur : {exc}\n")


if __name__ == "__main__":
    main()
