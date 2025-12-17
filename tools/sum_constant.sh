#!/bin/bash

# ==============================================================================
# summarize_results.sh
# 功能：解析 eval_sentence_level_stats_fast.py 的输出结果，生成终端可读的简报。
# 用法：./summarize_results.sh <结果目录>
# ==============================================================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# 1. 检查参数
DIR="$1"
if [ -z "$DIR" ]; then
    echo -e "${RED}Error: Please provide the result directory.${NC}"
    echo "Usage: $0 path/to/results"
    exit 1
fi

# 检查必要文件是否存在
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
# 2. 基础统计 (Sentence Stats)
# ------------------------------------------------------------------------------
if [ -f "$FILE_STATS" ]; then
    echo -e "${YELLOW}[1] Dataset Statistics (from sentence_stats.csv)${NC}"
    
    # 使用 awk 计算总数、平均时长、平均窗口数
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
# 3. 基线表现 ("I don't know")
# ------------------------------------------------------------------------------
if [ -f "$FILE_SPECIAL" ]; then
    echo -e "${YELLOW}[2] Fixed Baseline Performance ('I don't know')${NC}"
    # 提取 WER (列3) 和 BERTScore (列4) - 假设只有一行数据
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
# 4. 寻找最佳中心句 (Centroid)
# ------------------------------------------------------------------------------
echo -e "${YELLOW}[3] Best Centroid Sentence (Lowest WER against corpus)${NC}"

# Python 脚本生成的 CSV 已经是按 WER 排序的 (Ascending)
# 所以第2行 (NR==2) 就是最好的，最后一行 (tail) 是最差的。
# 注意：CSV格式为 sentence_id, text, WER, CER, BLEU, BERT
# 我们利用 awk $(NF-3) 取倒数第4列(WER)，防止 text 中有逗号干扰列数

# --- 最佳句子 (Top 1) ---
echo -e "${BOLD}>>> TOP 1 Candidate:${NC}"
head -n 2 "$FILE_MAIN" | tail -n 1 | awk -F, '{
    # 为了处理 text 中可能有逗号的情况，我们重新拼凑 text 字段
    # 假设 text 从第2个字段开始，到倒数第5个字段结束
    text=""
    for(i=2;i<=NF-4;i++) text = text $i ","
    # 去掉末尾多余逗号
    sub(/,$/, "", text)
    
    wer=$(NF-3)
    bert=$(NF)
    
    printf "  - Text: %s%s%s\n", "'"${GREEN}"'", text, "'"${NC}"'"
    printf "  - WER:  %s%.4f%s (Lower is better)\n", "'"${GREEN}"'", wer, "'"${NC}"'"
    printf "  - BERT: %s%.4f%s (Higher is better)\n", "'"${GREEN}"'", bert, "'"${NC}"'"
}'

echo ""
echo -e "${BOLD}>>> TOP 3 Candidates Overview:${NC}"
# 打印前3名的简表
head -n 4 "$FILE_MAIN" | awk -F, '
    BEGIN { printf "%-4s | %-10s | %-10s | %-s\n", "Rank", "WER", "BERT", "Text (Truncated)" }
    NR==1 { next }
    {
        # 简单处理文本截断
        txt=$2
        if(length(txt)>40) txt=substr(txt,1,37)"..."
        
        # 颜色逻辑
        val_wer=$(NF-3)
        val_bert=$(NF)
        
        printf "#%-3d | %.4f     | %.4f     | %s\n", NR-1, val_wer, val_bert, txt
    }
' | column -t

echo ""

# ------------------------------------------------------------------------------
# 5. 总结对比
# ------------------------------------------------------------------------------
echo -e "${YELLOW}[4] Improvement Summary${NC}"

# 获取 Baseline WER 和 Best WER 进行计算
BASE_WER=$(awk -F, 'NR==2 {print $3}' "$FILE_SPECIAL")
BEST_WER=$(head -n 2 "$FILE_MAIN" | tail -n 1 | awk -F, '{print $(NF-3)}')

if [ ! -z "$BASE_WER" ] && [ ! -z "$BEST_WER" ]; then
    IMPROVEMENT=$(awk -v b="$BASE_WER" -v n="$BEST_WER" 'BEGIN {print b - n}')
    echo -e "  Replacing 'I don't know' with the computed centroid reduces WER by: ${GREEN}${IMPROVEMENT}${NC}"
fi

echo -e "${BOLD}============================================================${NC}"