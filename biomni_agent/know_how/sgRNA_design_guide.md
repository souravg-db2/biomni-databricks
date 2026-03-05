# sgRNA Design Guide: Three-Tiered Approach

---

## Metadata

**Short Description**: Comprehensive guide for finding or designing sgRNAs using validated sequences, CRISPick datasets, or de novo design tools.

**Authors**: Biomni Team

**Version**: 1.0

**Last Updated**: November 2025

**License**: CC BY 4.0

**Commercial Use**: ✅ Allowed

## Citations and Acknowledgments

### If you use validated sgRNAs from our database (Option 1):
- **Database Source**: Addgene (https://www.addgene.org)
- **Citation**: Always cite the original publication associated with each sgRNA using the PubMed ID provided in the database
- **Acknowledgment**: "Validated sgRNA sequences obtained from Addgene (https://www.addgene.org/crispr/reference/grna-sequence/)"

### If you use CRISPick designs (Option 2):
- **Acknowledgment Statement**: "Guide designs provided by the CRISPick web tool of the GPP at the Broad Institute"
- **Citation for Cas9 designs (SpCas9, SaCas9)**: Sanson KR, et al. Optimized libraries for CRISPR-Cas9 genetic screens with multiple modalities. Nat Commun. 2018;9(1):5416. PMID: 30575746
- **Citation for Cas12a designs (AsCas12a, enAsCas12a)**: DeWeirdt PC, et al. Optimization of AsCas12a for combinatorial genetic screens in human cells. Nat Biotechnol. 2021;39(1):94-104. PMID: 32661438

---

## Overview

This guide provides a three-tiered approach to sgRNA design, prioritizing validated sequences before moving to computational predictions. Always start with Option 1 and proceed to subsequent options only if needed.

## Option 1: Search Validated sgRNA Sequences (Recommended First)

Use the available UC tools and literature search to find validated sgRNAs for your gene and species. Record the sgRNA sequence, reference (PubMed ID), and validation details.

## Option 2: Download Pre-Computed sgRNAs from CRISPick

When no validated sgRNAs are found, use CRISPick datasets for your organism and Cas enzyme. Match the dataset to your Cas variant (e.g. AsCas12a vs enAsCas12a). Filter by gene, then select top sgRNAs by Combined Rank. Use 3–4 sgRNAs per gene for validation.

## Option 3: General sgRNA Design Guidelines (Last Resort)

- **Length**: 20 bp for SpCas9/SaCas9; 23–25 bp for Cas12a.
- **PAM**: SpCas9 NGG; SaCas9 NNGRRT; Cas12a TTTV (5' of target).
- **GC content**: 40–60%. Avoid TTTT and long repeats.
- **Knockout**: Target early exons. **Activation**: -200 to +1 bp from TSS. **Inhibition**: -50 to +300 bp from TSS.

## Best Practices

- Always start with validated sequences (Option 1), then pre-computed designs (Option 2), then de novo (Option 3).
- Test 3–4 sgRNAs per target gene.
- Match CRISPick dataset to your Cas enzyme (e.g. enAsCas12a vs AsCas12a).
