"""
Brain region mapping for FACET-CV facial motor and speech behaviour analysis.

Maps pipeline screening indications (facial_paresis, buccofacial_apraxia,
dysarthria, speech_apraxia, phonological_disorder) to their underlying
neuroanatomical generators in the cortex, subcortex, brainstem, and cerebellum.

Each brain region entry in BRAIN_REGION_MAP carries:
  - Anatomical metadata (name, lobe, hemisphere, structure_type, pathway)
  - Normalised 2D coordinates for two schematic views consumed by
    Visualizer.plot_brain_activation_map:
      lat_xy: lateral left-hemisphere view (x=anterior to posterior,
              y=inferior to superior)
      sub_xy: brainstem/subcortical schematic panel
  - Clinical relevance text linking the region to the indication types

INDICATION_TO_REGIONS maps each indication_type string to the list of region
IDs that are implicated when that indication is reported. The mapping is
intentionally inclusive: all plausible neural substrates are listed so the
visualisation shows the full lesion hypothesis space rather than a single site.

Public API mirrors anatomy.py:
  BRAIN_REGION_MAP           -- region metadata dict
  INDICATION_TO_REGIONS      -- indication_type to List[region_id]
  map_findings_to_brain_regions(screening_results)
  aggregate_by_brain_region(screening_results)
  generate_brain_report(screening_results)

References
----------
Duffy JR (2013) Motor Speech Disorders: Substrates, Differential Diagnosis,
  and Management, 3rd ed. Elsevier Mosby, St Louis.
  Comprehensive reference for corticobulbar pathway anatomy, Brodmann area
  assignments (BA 4, BA 6, BA 44/45, BA 22), and clinical profiles of
  dysarthria subtypes used in BRAIN_REGION_MAP clinical_relevance fields.

Dronkers NF (1996) A new brain region for coordinating speech articulation.
  Nature 384(6605):159–161.
  Identified the left anterior insula as critical for speech apraxia
  (``insula_anterior`` entry); infarcts here produce apraxia of speech
  with preserved language comprehension.

Terao S, Miura N, Takatsu S, Mitsuma T, Takahashi A (2000) Clinical and
  neuropathological studies on the motor neuron lesion in patients with
  ALS.  J Neurol Sci 180(1–2):82–87.
  Corticobulbar degeneration pattern in ALS informing the dysarthria
  indicator mapping to primary motor cortex and corticobulbar tract.

Ackermann H, Riecker A (2004) The contribution of the insula to motor
  aspects of speech production: a review and a hypothesis. Brain Lang
  89(2):320–328.
  Insular role in orofacial motor sequencing; supports ``insula_anterior``
  and ``premotor_lateral`` entries for buccofacial_apraxia indication.

Murdoch BE (2010) The cerebellum and dysarthria. Int J Speech Lang Pathol
  12(6):480–483.
  Ataxic dysarthria profile and cerebellar contribution; basis for the
  ``cerebellum`` region entries and pathway = 'cerebellar' labels.

Lu J, Zhao L, Yang B, et al. (2021) Functional maps of direct electrical
  stimulation-induced speech arrest and anomia: a multicentre retrospective
  study. Brain 144(9):2566–2580.
  Speech arrest peaks at the ventral precentral gyrus (vPrCG), not Broca's
  area; anomia peaks at posterior STG and pars triangularis - defines the
  primary cortical substrates for the ``speech_motor_cortex`` /
  ``premotor_lateral`` / ``broca_area`` region entries.
  https://doi.org/10.1093/brain/awab125

Rossi M, Conti Nibali M, Viganò L, et al. (2021) Clinical pearls and
  methods for intraoperative motor mapping. Neurosurgery 88(5):949–960.
  Comprehensive reference for LF/HF DES protocols, orofacial EMG response
  detection, and the clinical distinction between speech arrest (motor
  cortex) and anomia (language cortex); supports the pathway labels and
  clinical_relevance fields for cortical motor region entries.
  https://doi.org/10.1093/neuros/nyaa359

Bello L, Gambini A, Castellano A, et al. (2014) Tailoring neurophysiological
  strategies with clinical context enhances resection safety and expands
  indications in gliomas involving motor pathways. Neuro-Oncology 16(5):748–763.
  LF/HF combined DES including multichannel EMG for orofacial muscles;
  corticobulbar tract anatomy underpinning the ``facial_upper`` /
  ``facial_lower`` innervation split and pathway = 'corticobulbar' labels.
  https://doi.org/10.1093/neuonc/not327

Collee E, Satoer D, Visch-Brink E, Hoeijmakers JGJ, Vincent AJPE (2022)
  Speech and Language Errors during Awake Brain Surgery and Postoperative
  Language Outcome in Glioma Patients. Cancers 14(21):5466.
  Intraoperative production errors (dysarthria/speech arrest) map to precentral
  gyrus; semantic errors to IFOF; phonemic errors to AF - provides the
  anatomy-to-indication mapping behind INDICATION_TO_REGIONS entries.
  https://doi.org/10.3390/cancers14215466

Pulvermüller F, Huss M, Kherif F, Moscoso del Prado Martin F, Hauk O,
  Shtyrov Y (2006) Motor cortex maps articulatory features of speech sounds.
  Proc Natl Acad Sci USA 103(20):7865–7870.
  Somatotopic motor cortex activation during speech - lip sounds activate
  ventrolateral M1, tongue sounds activate more lateral regions; supports
  the ``m1_face`` articulatory description and corticobulbar pathway entries.
  https://doi.org/10.1073/pnas.0509989103

Bress JN, Cascio CJ (2024) Sensorimotor regulation of facial expression:
  an untouched frontier. Neurosci Biobehav Rev 162:105684.
  Reviews sensorimotor cortex integration in volitional vs reflexive facial
  movements; distinguishes DES-accessible voluntary motor commands (M1,
  lateral premotor) from subcortical-emotional pathways - supports the
  clinical_relevance split between cortical and limbic / subcortical entries.
  https://doi.org/10.1016/j.neubiorev.2024.105684
"""

from typing import Any, Dict, List, Optional


BRAIN_REGION_MAP: Dict[str, Dict[str, Any]] = {
    "m1_face": {
        "name": "Primary Motor Cortex (face area)",
        "full_name": "Primary Motor Cortex, orofacial representation (Brodmann area 4)",
        "lobe": "frontal",
        "hemisphere": "contralateral (bilateral for upper face)",
        "structure_type": "cortical",
        "lat_xy": (0.44, 0.54),
        "sub_xy": (0.48, 0.78),
        "color_base": "#C0392B",
        "description": "Voluntary orofacial movement via corticobulbar projections to CN VII, IX, X, XII nuclei",
        "clinical_relevance": (
            "Lesion causes contralateral lower-face paresis; upper face spared "
            "due to bilateral cortical innervation of frontalis and orbicularis oculi"
        ),
        "pathway": "corticobulbar",
    },
    "premotor_lateral": {
        "name": "Lateral Premotor Cortex",
        "full_name": "Lateral Premotor / Ventral Premotor Cortex (Brodmann area 6, lateral)",
        "lobe": "frontal",
        "hemisphere": "left dominant",
        "structure_type": "cortical",
        "lat_xy": (0.32, 0.64),
        "sub_xy": (0.37, 0.80),
        "color_base": "#E67E22",
        "description": "Learned skilled orofacial movement sequences; motor program selection and online control",
        "clinical_relevance": (
            "Primary locus for buccofacial apraxia; lesion impairs volitional oral acts "
            "on command while reflexive movements remain intact"
        ),
        "pathway": "corticobulbar",
    },
    "sma": {
        "name": "Supplementary Motor Area",
        "full_name": "Supplementary Motor Area (Brodmann area 6, mesial surface)",
        "lobe": "frontal",
        "hemisphere": "bilateral",
        "structure_type": "cortical",
        "lat_xy": (0.42, 0.84),
        "sub_xy": (0.50, 0.83),
        "color_base": "#D35400",
        "description": "Self-initiated speech motor acts; sequential movement coordination",
        "clinical_relevance": (
            "SMA syndrome causes akinetic mutism; receives basal-ganglia loop "
            "input via thalamus; involved in initiation of volitional speech"
        ),
        "pathway": "corticobulbar",
    },
    "brocas_area": {
        "name": "Broca's Area",
        "full_name": "Broca's Area - IFG pars opercularis and triangularis (Brodmann areas 44/45)",
        "lobe": "frontal",
        "hemisphere": "left",
        "structure_type": "cortical",
        "lat_xy": (0.19, 0.51),
        "sub_xy": (0.31, 0.74),
        "color_base": "#8E44AD",
        "description": "Phonological encoding, syntactic processing, and articulatory motor planning",
        "clinical_relevance": (
            "Broca's aphasia (non-fluent, agrammatic speech); "
            "verbal and buccofacial apraxia; anatomically overlaps lateral premotor cortex"
        ),
        "pathway": "arcuate_fasciculus",
    },
    "anterior_insula": {
        "name": "Anterior Insula",
        "full_name": "Anterior Insula (Brodmann areas 13/14, agranular insular cortex)",
        "lobe": "insular",
        "hemisphere": "left dominant",
        "structure_type": "cortical",
        "lat_xy": (0.30, 0.47),
        "sub_xy": (0.40, 0.70),
        "color_base": "#2980B9",
        "description": "Sensorimotor integration for speech; articulatory sequencing and feedforward control",
        "clinical_relevance": (
            "Consistent lesion site for apraxia of speech; integrates "
            "somatosensory feedback with cortical motor commands"
        ),
        "pathway": "arcuate_fasciculus",
    },
    "wernickes_area": {
        "name": "Wernicke's Area",
        "full_name": "Wernicke's Area - posterior STG/STS (Brodmann areas 22/42)",
        "lobe": "temporal",
        "hemisphere": "left",
        "structure_type": "cortical",
        "lat_xy": (0.63, 0.38),
        "sub_xy": (0.65, 0.62),
        "color_base": "#27AE60",
        "description": "Phonological representation and speech perception; lexical access",
        "clinical_relevance": (
            "Wernicke's aphasia (fluent, paraphasic); "
            "lesion produces phonological paraphasias and impaired repetition"
        ),
        "pathway": "arcuate_fasciculus",
    },
    "angular_gyrus": {
        "name": "Angular Gyrus",
        "full_name": "Angular Gyrus (Brodmann area 39, inferior parietal lobule)",
        "lobe": "parietal",
        "hemisphere": "left",
        "structure_type": "cortical",
        "lat_xy": (0.69, 0.59),
        "sub_xy": (0.67, 0.67),
        "color_base": "#16A085",
        "description": "Multimodal integration; phonological awareness and semantic processing",
        "clinical_relevance": (
            "Phonological disorder and reading impairment; "
            "associated with phonological alexia and deep dyslexia"
        ),
        "pathway": "arcuate_fasciculus",
    },
    "supramarginal_gyrus": {
        "name": "Supramarginal Gyrus",
        "full_name": "Supramarginal Gyrus (Brodmann area 40, inferior parietal lobule)",
        "lobe": "parietal",
        "hemisphere": "left",
        "structure_type": "cortical",
        "lat_xy": (0.59, 0.61),
        "sub_xy": (0.59, 0.68),
        "color_base": "#1ABC9C",
        "description": "Phonological short-term memory; articulatory phonological encoding; phonological loop",
        "clinical_relevance": (
            "Lesion impairs phonological working memory and articulatory rehearsal; "
            "key node in non-word repetition"
        ),
        "pathway": "arcuate_fasciculus",
    },
    "basal_ganglia": {
        "name": "Basal Ganglia",
        "full_name": "Basal Ganglia - striatum and globus pallidus (caudate, putamen, GPi/GPe)",
        "lobe": "subcortical",
        "hemisphere": "bilateral",
        "structure_type": "subcortical",
        "lat_xy": (0.38, 0.54),
        "sub_xy": (0.50, 0.57),
        "color_base": "#F39C12",
        "description": "Motor program gating, amplitude scaling, and initiation; speech rhythm regulation",
        "clinical_relevance": (
            "Hypokinetic dysarthria in Parkinson disease; "
            "hyperkinetic dysarthria in Huntington disease"
        ),
        "pathway": "basal_ganglia_thalamo_cortical",
    },
    "cerebellum": {
        "name": "Cerebellum",
        "full_name": "Cerebellum - vermis and bilateral hemispheres (lobules V-VII)",
        "lobe": "posterior fossa",
        "hemisphere": "bilateral",
        "structure_type": "cerebellar",
        "lat_xy": (0.78, 0.17),
        "sub_xy": (0.73, 0.28),
        "color_base": "#E74C3C",
        "description": "Timing and coordination of articulatory movements; online error correction",
        "clinical_relevance": (
            "Ataxic dysarthria: irregular articulatory breakdown, "
            "excess equal stress, scanning speech, imprecise consonants"
        ),
        "pathway": "dentato_thalamo_cortical",
    },
    "pons_facial_nucleus": {
        "name": "Pontine Facial Nucleus (CN VII)",
        "full_name": "Facial Motor Nucleus, Pons (lower motor neuron, CN VII)",
        "lobe": "brainstem",
        "hemisphere": "ipsilateral",
        "structure_type": "brainstem",
        "lat_xy": (0.52, 0.20),
        "sub_xy": (0.50, 0.37),
        "color_base": "#C0392B",
        "description": "Lower motor neuron for all ipsilateral facial muscles (upper and lower face)",
        "clinical_relevance": (
            "Peripheral facial palsy: total ipsilateral facial paresis including forehead; "
            "Bell's palsy, acoustic neuroma, parotid tumour"
        ),
        "pathway": "peripheral_cn7",
    },
    "internal_capsule": {
        "name": "Internal Capsule",
        "full_name": "Internal Capsule - genu and posterior limb (corticobulbar/corticospinal fibres)",
        "lobe": "subcortical",
        "hemisphere": "contralateral",
        "structure_type": "subcortical",
        "lat_xy": (0.40, 0.50),
        "sub_xy": (0.48, 0.63),
        "color_base": "#7F8C8D",
        "description": "White matter conduit for corticobulbar and corticospinal projections to brainstem nuclei",
        "clinical_relevance": (
            "Capsular stroke: common cause of central facial paresis and spastic dysarthria; "
            "characteristic pure motor hemiplegia pattern"
        ),
        "pathway": "corticobulbar",
    },
}


INDICATION_TO_REGIONS: Dict[str, List[str]] = {
    "facial_paresis": [
        "m1_face",
        "internal_capsule",
        "pons_facial_nucleus",
    ],
    "buccofacial_apraxia": [
        "premotor_lateral",
        "brocas_area",
        "anterior_insula",
        "sma",
    ],
    "dysarthria": [
        "cerebellum",
        "basal_ganglia",
        "internal_capsule",
        "pons_facial_nucleus",
    ],
    "speech_apraxia": [
        "anterior_insula",
        "brocas_area",
        "premotor_lateral",
        "sma",
    ],
    "phonological_disorder": [
        "wernickes_area",
        "angular_gyrus",
        "supramarginal_gyrus",
        "brocas_area",
    ],
}


_SEVERITY_WEIGHTS: Dict[str, float] = {
    "mild": 0.40,
    "moderate": 0.70,
    "severe": 1.00,
}

_SEVERITY_RANK: Dict[str, int] = {
    "none": 0,
    "mild": 1,
    "moderate": 2,
    "severe": 3,
}

_PATHWAY_LABELS: Dict[str, str] = {
    "corticobulbar": "Corticobulbar Tract",
    "arcuate_fasciculus": "Arcuate Fasciculus / Dorsal Stream",
    "basal_ganglia_thalamo_cortical": "Basal Ganglia - Thalamo-Cortical Loop",
    "dentato_thalamo_cortical": "Dentato-Thalamo-Cortical Tract",
    "peripheral_cn7": "CN VII (Peripheral - Facial Nerve)",
}

_MNI_COORDS: Dict[str, List[tuple]] = {
    "m1_face":             [(-50, -10,  45)],
    "premotor_lateral":    [(-52,   4,  42)],
    "sma":                 [( -4,  -4,  62)],
    "brocas_area":         [(-48,  20,   6)],
    "anterior_insula":     [(-38,  14,   2)],
    "wernickes_area":      [(-57, -42,  16)],
    "angular_gyrus":       [(-47, -65,  34)],
    "supramarginal_gyrus": [(-56, -44,  44)],
    "basal_ganglia":       [(-22,   8,   4), ( 22,   8,   4)],
    "cerebellum":          [(-20, -62, -26), ( 20, -62, -26)],
    "pons_facial_nucleus": [(  0, -30, -38)],
    "internal_capsule":    [(-22,  -4,   8), ( 22,  -4,   8)],
}


def map_findings_to_brain_regions(
    screening_results: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Map screening indications to per-region activation levels.

    Iterates over all indications in screening_results['indications'],
    resolves the implicated brain regions via INDICATION_TO_REGIONS, and
    computes a scalar activation score (0–1) for each region as:

        activation = max over all implying indications of
                     severity_weight(severity) × confidence

    where severity_weight is 0.4 / 0.7 / 1.0 for mild / moderate / severe.
    A region implicated by multiple indications receives the maximum
    activation, not a sum, to avoid double-counting.

    Returns a dict keyed by region_id. Every region in BRAIN_REGION_MAP is
    present; regions not implicated by any finding have activation = 0.0.
    Each value merges the static BRAIN_REGION_MAP metadata with:
      activation     - float in [0, 1]
      indications    - List[str] of indication_types that implicate this region
      max_severity   - str: highest severity among implying indications
      max_confidence - float: highest confidence among implying indications
    """
    indications = screening_results.get("indications", [])

    raw_scores: Dict[str, Dict[str, Any]] = {}
    for ind in indications:
        ind_type = ind.get("indication_type", "")
        severity = ind.get("severity", "mild")
        confidence = float(ind.get("confidence", 0.5))
        weight = _SEVERITY_WEIGHTS.get(severity, 0.40) * confidence

        for region_id in INDICATION_TO_REGIONS.get(ind_type, []):
            if region_id not in raw_scores:
                raw_scores[region_id] = {
                    "activation": 0.0,
                    "indications": [],
                    "max_severity": "none",
                    "max_confidence": 0.0,
                }
            entry = raw_scores[region_id]
            entry["activation"] = max(entry["activation"], weight)
            if ind_type not in entry["indications"]:
                entry["indications"].append(ind_type)
            if _SEVERITY_RANK.get(severity, 0) > _SEVERITY_RANK.get(entry["max_severity"], 0):
                entry["max_severity"] = severity
            entry["max_confidence"] = max(entry["max_confidence"], confidence)

    result: Dict[str, Dict[str, Any]] = {}
    for region_id, region_meta in BRAIN_REGION_MAP.items():
        scores = raw_scores.get(region_id, {})
        result[region_id] = {
            **region_meta,
            "activation": scores.get("activation", 0.0),
            "indications": scores.get("indications", []),
            "max_severity": scores.get("max_severity", "none"),
            "max_confidence": scores.get("max_confidence", 0.0),
        }

    return result


def aggregate_by_brain_region(
    screening_results: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Return activated brain regions sorted by activation descending.

    Thin wrapper around map_findings_to_brain_regions that filters to
    regions whose activation > 0 and returns them in descending order.
    """
    all_regions = map_findings_to_brain_regions(screening_results)
    activated = {r: v for r, v in all_regions.items() if v["activation"] > 0.0}
    return dict(sorted(activated.items(), key=lambda x: x[1]["activation"], reverse=True))


def generate_brain_report(
    screening_results: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate a structured neuroanatomical report from screening results.

    Identifies activated brain regions, groups them by structure type, determines
    which neural pathways are implicated, and produces a plain-text clinical
    localisation note based on the combination of active regions.

    Returns a dict with the following keys:
      regions_activated    - List[str] of activated region IDs (descending activation)
      activation_map       - Dict[region_id -> full activation info]
      by_structure_type    - Dict[structure_type -> List[region_id]]
      pathways_involved    - List[str] of unique pathway labels for activated regions
      n_regions_activated  - int
      clinical_localisation - str: plain-text localisation summary
      laterality_note      - str: hemisphere predominance note
    """
    activation_map = map_findings_to_brain_regions(screening_results)
    activated = {r: v for r, v in activation_map.items() if v["activation"] > 0.0}

    by_type: Dict[str, List[str]] = {}
    pathways_seen: List[str] = []
    for r_id, r_info in activated.items():
        stype = r_info.get("structure_type", "cortical")
        by_type.setdefault(stype, []).append(r_id)
        pw_key = r_info.get("pathway", "")
        pw_label = _PATHWAY_LABELS.get(pw_key, pw_key)
        if pw_label and pw_label not in pathways_seen:
            pathways_seen.append(pw_label)

    indication_types = {
        ind.get("indication_type", "")
        for ind in screening_results.get("indications", [])
    }

    notes: List[str] = []
    if "facial_paresis" in indication_types:
        if "pons_facial_nucleus" in activated and "m1_face" not in activated:
            notes.append(
                "Peripheral facial palsy pattern: pontine nucleus or distal CN VII"
            )
        elif "m1_face" in activated or "internal_capsule" in activated:
            notes.append(
                "Central facial paresis: lesion above the facial nucleus, "
                "cortex or internal capsule"
            )
        else:
            notes.append(
                "Facial paresis: localisation requires additional clinical context"
            )

    if "buccofacial_apraxia" in indication_types or "speech_apraxia" in indication_types:
        notes.append(
            "Praxic deficit pattern: left frontal operculum (Broca/premotor) "
            "and anterior insula are primary regions of interest"
        )

    if "dysarthria" in indication_types:
        if "cerebellum" in activated:
            notes.append(
                "Cerebellar circuit involvement - consider ataxic dysarthria"
            )
        if "basal_ganglia" in activated:
            notes.append(
                "Basal ganglia circuit involvement - consider hypo- or hyperkinetic dysarthria"
            )
        if "internal_capsule" in activated:
            notes.append(
                "Corticobulbar tract involvement - consider spastic dysarthria component"
            )

    if "phonological_disorder" in indication_types:
        notes.append(
            "Posterior language network (Wernicke's area, IPL): "
            "phonological processing impairment"
        )

    left_dominant = any(
        BRAIN_REGION_MAP.get(r, {}).get("hemisphere", "") in ("left", "left dominant")
        for r in activated
    )
    laterality_note = (
        "Left hemisphere (language-dominant) regions predominantly implicated"
        if left_dominant
        else "Bilateral or right-dominant pattern"
    )

    return {
        "regions_activated": list(
            sorted(activated, key=lambda r: activated[r]["activation"], reverse=True)
        ),
        "activation_map": activation_map,
        "by_structure_type": by_type,
        "pathways_involved": pathways_seen,
        "n_regions_activated": len(activated),
        "clinical_localisation": " | ".join(notes) if notes else
            "No specific localisation pattern identified",
        "laterality_note": laterality_note,
    }
