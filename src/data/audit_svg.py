"""
SVG Audit Script — Phase 1, Step 1

PURPOSE:
    Empirically discover the XML structure of CubiCasa5k SVG annotation
    files BEFORE writing any annotation conversion code. This script
    answers the critical question: "What XML tags and attributes represent
    doors, windows, walls, and other architectural elements?"

OUTPUT:
    - Console summary of findings
    - docs/svg_audit_report.txt — permanent record for reference
    - docs/svg_sample_structure.xml — a pretty-printed example SVG

HOW TO RUN:
    make audit-svg
    OR
    python src/data/audit_svg.py
"""

import random
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

# ── Allow imports from project root ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import load_config, load_env, get_env
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
CATEGORIES = ["colorful", "high_quality", "high_quality_architectural"]
SAMPLES_PER_CATEGORY = 5   # How many SVG files to examine per category
RANDOM_SEED = 42


def strip_namespace(tag: str) -> str:
    """
    Remove XML namespace prefix from a tag.

    SVG elements often look like: {http://www.w3.org/2000/svg}rect
    This function returns just: rect

    Why: When iterating XML with ElementTree, namespace URIs are prepended
    to every tag name. Stripping them makes tag comparison straightforward.
    """
    return tag.split("}")[-1] if "}" in tag else tag


def get_all_namespaces(svg_path: Path) -> dict[str, str]:
    """
    Extract all XML namespace declarations from the SVG file.

    Why: Knowing the namespaces tells us if the SVG uses standard SVG
    elements, custom namespaces, or mixed. This affects how we parse it.

    Returns dict of {prefix: uri}, e.g. {'svg': 'http://www.w3.org/2000/svg'}
    """
    namespaces = {}
    # ET.iterparse fires events as it reads the file token by token
    for event, elem in ET.iterparse(svg_path, events=["start-ns"]):
        prefix, uri = elem
        namespaces[prefix] = uri
    return namespaces


def audit_single_svg(svg_path: Path) -> dict:
    """
    Perform a deep audit of one SVG file.

    Collects:
    - All unique tag names (stripped of namespace)
    - All attribute names and their unique values (per tag)
    - Depth of the XML tree (how deeply nested elements go)
    - Count of each tag

    Returns a structured dictionary of findings.
    """
    findings = {
        "path": str(svg_path),
        "namespaces": {},
        "tag_counts": Counter(),
        "tag_attributes": defaultdict(lambda: defaultdict(set)),
        "max_depth": 0,
        "total_elements": 0,
    }

    try:
        findings["namespaces"] = get_all_namespaces(svg_path)
        tree = ET.parse(svg_path)
        root = tree.getroot()

        # Recursive traversal to capture tree depth and all elements
        def traverse(element, depth=0):
            findings["max_depth"] = max(findings["max_depth"], depth)
            findings["total_elements"] += 1

            tag = strip_namespace(element.tag)
            findings["tag_counts"][tag] += 1

            # Record every attribute name and its value for this tag
            for attr_name, attr_value in element.attrib.items():
                clean_attr = strip_namespace(attr_name)
                # For long values (like 'points' with hundreds of coords),
                # store a truncated preview instead of the full value
                if len(attr_value) > 80:
                    preview = attr_value[:80] + "... [truncated]"
                else:
                    preview = attr_value
                findings["tag_attributes"][tag][clean_attr].add(preview)

            for child in element:
                traverse(child, depth + 1)

        traverse(root)

    except ET.ParseError as e:
        logger.error(f"Failed to parse {svg_path}: {e}")
        findings["error"] = str(e)

    return findings


def collect_svg_samples(cubicasa_root: Path, n: int = SAMPLES_PER_CATEGORY) -> dict[str, list[Path]]:
    """
    Collect a random sample of SVG file paths from each category.

    Why random.seed: Ensures the same files are sampled every time
    this script is run, making the audit reproducible.
    """
    random.seed(RANDOM_SEED)
    samples = {}

    for category in CATEGORIES:
        cat_path = cubicasa_root / category
        if not cat_path.exists():
            logger.warning(f"Category folder not found: {cat_path}")
            samples[category] = []
            continue

        # Find all model.svg files within this category
        svg_files = sorted(cat_path.rglob("model.svg"))

        if not svg_files:
            logger.warning(f"No SVG files found in {cat_path}")
            samples[category] = []
            continue

        # Sample min(n, available) files
        count = min(n, len(svg_files))
        samples[category] = random.sample(svg_files, count)
        logger.info(f"{category}: sampling {count} of {len(svg_files)} SVG files")

    return samples


def format_report(all_findings: dict[str, list[dict]]) -> str:
    """
    Format the complete audit findings into a readable report string.
    This report is saved to docs/ and becomes a permanent reference
    document for the annotation conversion phase.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("CubiCasa5k SVG AUDIT REPORT")
    lines.append("Generated by: src/data/audit_svg.py")
    lines.append("=" * 70)
    lines.append("")

    # ── Per-category findings ─────────────────────────────────────────────
    for category, findings_list in all_findings.items():
        lines.append(f"\n{'─' * 70}")
        lines.append(f"CATEGORY: {category.upper()}")
        lines.append(f"{'─' * 70}")

        if not findings_list:
            lines.append("  No SVG files found or sampled.")
            continue

        # Aggregate across all samples in this category
        agg_tag_counts = Counter()
        agg_tag_attrs = defaultdict(lambda: defaultdict(set))
        agg_namespaces = set()

        for f in findings_list:
            if "error" in f:
                lines.append(f"  ERROR in {f['path']}: {f['error']}")
                continue
            agg_tag_counts.update(f["tag_counts"])
            for tag, attrs in f["tag_attributes"].items():
                for attr, values in attrs.items():
                    agg_tag_attrs[tag][attr].update(values)
            for prefix, uri in f["namespaces"].items():
                agg_namespaces.add(f"{prefix!r}: {uri}")

        # Namespaces
        lines.append("\n  XML Namespaces found:")
        for ns in sorted(agg_namespaces):
            lines.append(f"    {ns}")

        # Tag counts (sorted by frequency)
        lines.append("\n  XML Tags (by frequency):")
        for tag, count in agg_tag_counts.most_common():
            lines.append(f"    <{tag}>  ×{count}")

        # Detailed attributes per tag
        lines.append("\n  Attributes per Tag:")
        for tag in sorted(agg_tag_attrs.keys()):
            lines.append(f"\n    <{tag}>")
            for attr, values in sorted(agg_tag_attrs[tag].items()):
                # Show up to 8 unique values as examples
                sample_values = sorted(list(values))[:8]
                values_str = " | ".join(f'"{v}"' for v in sample_values)
                if len(values) > 8:
                    values_str += f" ... ({len(values)} unique values)"
                lines.append(f"      {attr}: {values_str}")

    # ── What to look for ─────────────────────────────────────────────────
    lines.append(f"\n{'=' * 70}")
    lines.append("WHAT TO LOOK FOR IN THIS REPORT")
    lines.append("=" * 70)
    lines.append("""
After reading this report, you need to identify:

1. WHICH TAG represents architectural elements?
   Look for tags like: rect, polygon, polyline, path, g, use
   The most frequent non-structural tags are likely the element containers.

2. WHICH ATTRIBUTE encodes the element TYPE (door/window/wall)?
   Common patterns:
   - class="Door"        → attribute name is 'class'
   - id="door_123"       → attribute name is 'id' (need prefix matching)
   - type="window"       → attribute name is 'type'
   - label="Staircase"   → attribute name is 'label'

3. WHAT ARE THE EXACT STRING VALUES for each class?
   e.g. class attribute might use "Door", "door", "DOOR", or "Opening"
   Document these exactly — they go into configs/dataset.yaml

4. HOW ARE COORDINATES ENCODED?
   - rect: uses x, y, width, height attributes
   - polygon/polyline: uses 'points' attribute (space-separated x,y pairs)
   - path: uses 'd' attribute (SVG path commands — most complex to parse)

5. ARE THERE TRANSFORM ATTRIBUTES?
   If elements have transform="matrix(...)" or transform="translate(...)",
   you must apply these transforms to get correct pixel coordinates.
   This is a common source of annotation errors.

Document your answers in docs/svg_findings.md after running this script.
""")

    return "\n".join(lines)


def save_sample_xml(svg_path: Path, output_path: Path) -> None:
    """
    Save a pretty-printed version of one SVG file for manual inspection.

    Why: ElementTree doesn't preserve the original formatting. This gives
    you a consistently indented XML file that's easy to read in an editor
    or browser.
    """
    try:
        tree = ET.parse(svg_path)
        ET.indent(tree, space="  ")  # Python 3.9+
        tree.write(output_path, encoding="unicode", xml_declaration=True)
        logger.info(f"Sample SVG saved to: {output_path}")
    except Exception as e:
        logger.error(f"Could not save sample XML: {e}")


def main():
    load_env()

    cubicasa_root_str = get_env("CUBICASA_ROOT")
    if not cubicasa_root_str:
        logger.error(
            "CUBICASA_ROOT not set in .env file.\n"
            "Edit your .env file and set: CUBICASA_ROOT=path/to/cubicasa5k"
        )
        sys.exit(1)

    cubicasa_root = Path(cubicasa_root_str)
    if not cubicasa_root.exists():
        logger.error(f"Dataset path does not exist: {cubicasa_root}")
        sys.exit(1)

    logger.info(f"Auditing CubiCasa5k dataset at: {cubicasa_root}")
    logger.info(f"Sampling {SAMPLES_PER_CATEGORY} SVG files per category")

    # ── Collect samples ───────────────────────────────────────────────────
    samples = collect_svg_samples(cubicasa_root, n=SAMPLES_PER_CATEGORY)

    # ── Audit each sample ─────────────────────────────────────────────────
    all_findings = {}
    for category, svg_paths in samples.items():
        logger.info(f"Auditing category: {category}")
        findings = []
        for svg_path in svg_paths:
            logger.debug(f"  Auditing: {svg_path}")
            findings.append(audit_single_svg(svg_path))
        all_findings[category] = findings

    # ── Generate and save report ──────────────────────────────────────────
    report = format_report(all_findings)
    print(report)

    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)

    report_path = docs_dir / "svg_audit_report.txt"
    report_path.write_text(report, encoding="utf-8")
    logger.info(f"Audit report saved to: {report_path}")

    # ── Save one pretty-printed SVG example per category ─────────────────
    for category, svg_paths in samples.items():
        if svg_paths:
            output_path = docs_dir / f"svg_sample_{category}.xml"
            save_sample_xml(svg_paths[0], output_path)

    logger.info("Audit complete. Next steps:")
    logger.info("1. Open docs/svg_audit_report.txt and read the findings.")
    logger.info("2. Open docs/svg_sample_colorful.xml in a text editor.")
    logger.info("3. Fill in docs/svg_findings.md with your conclusions.")
    logger.info("4. Update configs/dataset.yaml with the correct svg_class_tags.")


if __name__ == "__main__":
    main()