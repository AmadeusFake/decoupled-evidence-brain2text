#!/bin/bash

# ==============================================================================
# summarize_results.sh
# Purpose: Parse the outputs of eval_sentence_level_stats_fast.py and print a
#          terminal-friendly summary report.
# Usage:   ./summarize_results.sh <results_dir>
# ==============================================================================

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# 1) Argument check
DIR="$1"
if [ -z "$DIR" ]; then
    echo -e "${RED}Error: Please provide the result directory.${NC}"
    echo "Usage: $0 path/to/results"
    exit 1
fi

# Check required files
FILE_MAIN="$DIR/all_sentences_as_baseline.csv"
FILE_SPECIAL="$DIR/special_baseline_idontknow.csv"
FILE_STATS="$DIR/sentence_stats.csv"

if [ ! -f "$FILE_MAIN" ]; then
    echo -e "${RED}Error: $FILE_MAIN not found.${NC}"
    exit 1
fi

echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD}             Evaluation Summary Report                      ${NC}"
echo -e "${BOLD}============================================================${NC}"
echo -e "Directory: ${CYAN}$DIR${NC}"
echo ""

# ------------------------------------------------------------------------------
# 2) Basic statistics (Sentence Stats)
# ------------------------------------------------------------------------------
if [ -f "$FILE_STATS" ]; then
    echo -e "${YELLOW}[1] Dataset Statistics (from sentence_stats.csv)${NC}"

    # Use awk to compute total count, average duration, and average window count
    awk -F, '
        NR>1 {
            count++;
            sum_dur+=$3;
            sum_win+=$4
        }
        END {
            if (count > 0) {
                printf "  - Total Sentences: %d\n", count
                printf "  - Avg Duration:    %.2f s\n", sum_dur/count
                printf "  - Avg Windows:     %.2f\n", sum_win/count
            } else {
                print "  - No data found."
            }
        }
    ' "$FILE_STATS"
else
    echo -e "${RED}[!] sentence_stats.csv not found.${NC}"
fi
echo ""

# ------------------------------------------------------------------------------
# 3) Fixed baseline performance ("I don't know")
# ------------------------------------------------------------------------------
if [ -f "$FILE_SPECIAL" ]; then
    echo -e "${YELLOW}[2] Fixed Baseline Performance ('I don't know')${NC}"
    # Extract WER (col 3) and BERTScore (col 4) — assume exactly one data row
    awk -F, 'NR==2 {
        printf "  - Text: %s\n", $1
        printf "  - WER:  %s%.4f%s\n", "'"${RED}"'", $3, "'"${NC}"'"
        printf "  - BERT: %s%.4f%s\n", "'"${RED}"'", $4, "'"${NC}"'"
    }' "$FILE_SPECIAL"
else
    echo -e "  - Special baseline file not found."
fi
echo ""

# ------------------------------------------------------------------------------
# 4) Find the best centroid sentence (Centroid)
# ------------------------------------------------------------------------------
echo -e "${YELLOW}[3] Best Centroid Sentence (Lowest WER against corpus)${NC}"

# The Python-generated CSV is already sorted by WER (ascending),
# so line 2 (NR==2) is the best and the last line is the worst.
# CSV format: sentence_id, text, WER, CER, BLEU, BERT
# We use $(NF-3) to pick the 4th field from the end (WER) to avoid issues
# when the text field contains commas.

# --- Best sentence (Top 1) ---
echo -e "${BOLD}>>> TOP 1 Candidate:${NC}"
head -n 2 "$FILE_MAIN" | tail -n 1 | awk -F, '{
    # To handle possible commas in the text, reconstruct the text field:
    # Assume text spans from field 2 to field (NF-4)
    text=""
    for(i=2;i<=NF-4;i++) text = text $i ","
    # Remove trailing comma
    sub(/,$/, "", text)

    wer=$(NF-3)
    bert=$(NF)

    printf "  - Text: %s%s%s\n", "'"${GREEN}"'", text, "'"${NC}"'"
    printf "  - WER:  %s%.4f%s (Lower is better)\n", "'"${GREEN}"'", wer, "'"${NC}"'"
    printf "  - BERT: %s%.4f%s (Higher is better)\n", "'"${GREEN}"'", bert, "'"${NC}"'"
}'

echo ""
echo -e "${BOLD}>>> TOP 3 Candidates Overview:${NC}"
# Print a compact table for the top-3 candidates
head -n 4 "$FILE_MAIN" | awk -F, '
    BEGIN { printf "%-4s | %-10s | %-10s | %-s\n", "Rank", "WER", "BERT", "Text (Truncated)" }
    NR==1 { next }
    {
        # Simple truncation
        txt=$2
        if(length(txt)>40) txt=substr(txt,1,37)"..."

        val_wer=$(NF-3)
        val_bert=$(NF)

        printf "#%-3d | %.4f     | %.4f     | %s\n", NR-1, val_wer, val_bert, txt
    }
' | column -t

echo ""

# ------------------------------------------------------------------------------
# 5) Summary comparison
# ------------------------------------------------------------------------------
echo -e "${YELLOW}[4] Improvement Summary${NC}"

# Compute baseline WER vs best WER
BASE_WER=$(awk -F, 'NR==2 {print $3}' "$FILE_SPECIAL")
BEST_WER=$(head -n 2 "$FILE_MAIN" | tail -n 1 | awk -F, '{print $(NF-3)}')

if [ ! -z "$BASE_WER" ] && [ ! -z "$BEST_WER" ]; then
    IMPROVEMENT=$(awk -v b="$BASE_WER" -v n="$BEST_WER" 'BEGIN {print b - n}')
    echo -e "  Replacing 'I don't know' with the computed centroid reduces WER by: ${GREEN}${IMPROVEMENT}${NC}"
fi

echo -e "${BOLD}============================================================${NC}"
