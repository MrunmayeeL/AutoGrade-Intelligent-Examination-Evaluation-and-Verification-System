import os
import re
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional

logger = logging.getLogger("HandwrittenPipeline.Evaluation")

def run_evaluation_metrics(
    results_csv_path: str,
    output_dir: str,
    ground_truth_csv_path: Optional[str] = None
) -> None:
    """
    Computes statistical evaluation metrics and saves plots comparing predicted grades against teacher marks.

    Args:
        results_csv_path: Path to the generated batch_results.csv.
        output_dir: Directory where plots and summaries should be saved.
        ground_truth_csv_path: Path to a CSV containing ground truth teacher marks.
                               If provided, we try to merge teacher marks for evaluation.
    """
    logger.info(f"Loading results from {results_csv_path}...")
    if not os.path.exists(results_csv_path):
        logger.error(f"Results CSV not found: {results_csv_path}. Skipping evaluation.")
        return
        
    df = pd.read_csv(results_csv_path)
    if df.empty:
        logger.warning("Results CSV is empty. Skipping evaluation.")
        return
        
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Merge Ground Truth if specified and not already present
    if "teacher_marks" not in df.columns and ground_truth_csv_path:
        if os.path.exists(ground_truth_csv_path):
            logger.info(f"Merging ground truth marks from {ground_truth_csv_path}...")
            try:
                gt_df = pd.read_csv(ground_truth_csv_path)
                
                # We can match on normalized questions and similar student answers if exact student_id is not present
                # Or if the ground truth file has a structure we join on. Let's do a best-effort join
                # Let's inspect ground truth column names and structure
                if "SID" in gt_df.columns:
                    logger.info("Detected Gradescope-style scores sheet. Mapping SID and question columns...")
                    teacher_marks_list = []
                    
                    for _, row in df.iterrows():
                        # Extract and clean student ID
                        student_id = str(row["student_id"]).replace("student_", "").strip()
                        # Clean question code
                        question_code = str(row["question"]).replace("Q", "").strip()
                        
                        # Find matching student row
                        student_match = gt_df[gt_df["SID"].astype(str).str.strip() == student_id]
                        
                        if not student_match.empty:
                            # Search for column that corresponds to the question code (e.g. contains 'Q1')
                            q_col = None
                            for col in gt_df.columns:
                                # Match columns like '1: Q1 (12.0 pts)' or 'Q1'
                                if re.search(r'\bQ' + re.escape(question_code) + r'\b', col, re.IGNORECASE):
                                    q_col = col
                                    break
                            
                            if q_col:
                                score = student_match.iloc[0][q_col]
                                teacher_marks_list.append(float(score))
                            else:
                                logger.warning(f"Could not find Gradescope score column for question Q{question_code}")
                                teacher_marks_list.append(np.nan)
                        else:
                            logger.warning(f"Could not find student {student_id} in ground truth scores sheet")
                            teacher_marks_list.append(np.nan)
                    
                    df["teacher_marks"] = teacher_marks_list
                elif "questions" in gt_df.columns and "student_answer" in gt_df.columns:
                    # Rename columns to match results format
                    gt_df = gt_df.rename(columns={"questions": "question_raw", "student_answer": "student_answer_raw"})
                    
                    # Create normalized matching keys
                    df["q_norm"] = df["question"].str.replace(r'[\.\)\s]', '', regex=True).str.upper()
                    gt_df["q_norm"] = gt_df["q_norm"] = gt_df["question_raw"].astype(str).str.replace(r'[\.\)\s]', '', regex=True).str.upper()
                    
                    # Merge on normalized question
                    if len(df) == len(gt_df):
                        logger.info("Batch results size matches ground truth dataset size. Mapping rows sequentially.")
                        df["teacher_marks"] = gt_df["teacher_marks"].values
                    else:
                        logger.info("Attempting fuzzy match on student answers...")
                        # Map using text prefix matching
                        teacher_marks_list = []
                        for _, row in df.iterrows():
                            ans = str(row["student_answer"])[:30].lower().strip()
                            # Find matching answer in ground truth
                            match = gt_df[gt_df["student_answer_raw"].astype(str).str.lower().str.strip().str.startswith(ans)]
                            if not match.empty:
                                teacher_marks_list.append(float(match.iloc[0]["teacher_marks"]))
                            else:
                                teacher_marks_list.append(np.nan)
                        df["teacher_marks"] = teacher_marks_list
            except Exception as e:
                logger.error(f"Failed to merge ground truth: {e}", exc_info=True)
        else:
            logger.warning(f"Ground truth CSV path '{ground_truth_csv_path}' does not exist.")

    # 2. Check if we have teacher marks to run quantitative evaluation
    has_teacher_marks = "teacher_marks" in df.columns and df["teacher_marks"].notna().any()
    
    # Fill missing values
    df["llm_grade"] = pd.to_numeric(df["llm_grade"], errors="coerce").fillna(0.0)
    df["sbert_score"] = pd.to_numeric(df["sbert_score"], errors="coerce").fillna(0.0)
    df["bert_score"] = pd.to_numeric(df["bert_score"], errors="coerce").fillna(0.0)
    
    if has_teacher_marks:
        df["teacher_marks"] = pd.to_numeric(df["teacher_marks"], errors="coerce").fillna(0.0)
        df["total_marks"] = pd.to_numeric(df.get("max_marks", 5), errors="coerce").fillna(5.0)
        
        # Calculate stats
        from scipy.stats import pearsonr, spearmanr
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        mae = mean_absolute_error(df["teacher_marks"], df["llm_grade"])
        mse = mean_squared_error(df["teacher_marks"], df["llm_grade"])
        rmse = np.sqrt(mse)
        
        # Pearson and Spearman correlations
        p_corr, _ = pearsonr(df["teacher_marks"], df["llm_grade"]) if len(df) > 1 else (0.0, 1.0)
        s_corr, _ = spearmanr(df["teacher_marks"], df["llm_grade"]) if len(df) > 1 else (0.0, 1.0)
        
        sbert_corr, _ = pearsonr(df["teacher_marks"], df["sbert_score"]) if len(df) > 1 else (0.0, 1.0)
        bert_corr, _ = pearsonr(df["teacher_marks"], df["bert_score"]) if len(df) > 1 else (0.0, 1.0)
        
        # Output summary report
        summary_path = os.path.join(output_dir, "evaluation_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("END-TO-END HANDWRITTEN GRADING PIPELINE EVALUATION REPORT\n")
            f.write("=========================================================\n")
            f.write(f"Sample Size Evaluated: {len(df)}\n")
            f.write(f"Mean Absolute Error (MAE):     {mae:.4f}\n")
            f.write(f"Mean Squared Error (MSE):      {mse:.4f}\n")
            f.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")
            f.write(f"Pearson Correlation (Grade vs Ground Truth):  {p_corr:.4f}\n")
            f.write(f"Spearman Correlation (Grade vs Ground Truth): {s_corr:.4f}\n")
            f.write(f"SBERT Score Correlation with Teacher Marks:   {sbert_corr:.4f}\n")
            f.write(f"BERT Score Correlation with Teacher Marks:    {bert_corr:.4f}\n")
            f.write("=========================================================\n")
            
        logger.info(f"Evaluation report saved to: {summary_path}")
        
        # Plotting Setup
        sns.set_theme(style="whitegrid")
        max_mark = df["total_marks"].max()
        
        # Plot 1: Scatter plot of Grade Correlation
        plt.figure(figsize=(8, 6))
        sns.regplot(
            x="teacher_marks", 
            y="llm_grade", 
            data=df, 
            color="#2c3e50", 
            scatter_kws={"s": 80, "alpha": 0.7, "color": "#16a085"},
            line_kws={"color": "#e74c3c", "linewidth": 2}
        )
        plt.plot([0, max_mark], [0, max_mark], linestyle="--", color="gray", alpha=0.5, label="Perfect Agreement")
        plt.title("LLM Assigned Grades vs. Teacher Marks (Handwritten)", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Teacher Marks (Ground Truth)", fontsize=12)
        plt.ylabel("LLM Assigned Grade", fontsize=12)
        plt.xlim(-0.5, max_mark + 0.5)
        plt.ylim(-0.5, max_mark + 0.5)
        plt.text(0.1, max_mark - 0.5, f"MAE: {mae:.2f}\nPearson r: {p_corr:.2f}\nSpearman ρ: {s_corr:.2f}", 
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'), fontsize=10)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "grade_correlation_scatter.png"), dpi=300)
        plt.close()
        
        # Plot 2: Overlapping Distribution / KDE plot
        plt.figure(figsize=(8, 6))
        sns.kdeplot(df["teacher_marks"], fill=True, color="#3498db", label="Teacher Marks (Ground Truth)", alpha=0.4, bw_adjust=0.8)
        sns.kdeplot(df["llm_grade"], fill=True, color="#2ecc71", label="LLM Assigned Grade", alpha=0.4, bw_adjust=0.8)
        plt.title("Distribution of Assigned Grades", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Marks / Grade", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "grade_distribution_kde.png"), dpi=300)
        plt.close()
        
        # Plot 3: Box plot of Similarity Scores vs. Teacher Marks
        plt.figure(figsize=(10, 5))
        melted_df = df.melt(
            id_vars=["teacher_marks"], 
            value_vars=["sbert_score", "bert_score"],
            var_name="Score Type", 
            value_name="Similarity Score"
        )
        melted_df["Score Type"] = melted_df["Score Type"].replace({"sbert_score": "SBERT", "bert_score": "BERTScore"})
        
        sns.boxplot(
            x="teacher_marks", 
            y="Similarity Score", 
            hue="Score Type", 
            data=melted_df,
            palette={"SBERT": "#34495e", "BERTScore": "#e67e22"}
        )
        plt.title("Semantic Similarity Scores Grouped by Teacher Marks", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Teacher Marks (Ground Truth)", fontsize=12)
        plt.ylabel("Semantic Similarity Score", fontsize=12)
        plt.ylim(0, 1.05)
        plt.legend(title="Algorithm", frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "similarity_scores_vs_marks.png"), dpi=300)
        plt.close()

        # Plot 4: Average Grades comparison
        plt.figure(figsize=(6, 6))
        avg_teacher = df["teacher_marks"].mean()
        avg_ai = df["llm_grade"].mean()
        bars = plt.bar(["Teacher's Average", "AI's Average"], [avg_teacher, avg_ai], color=['#3498db', '#2ecc71'], width=0.5)
        plt.title("Average Score Comparison", fontsize=14, fontweight="bold", pad=15)
        plt.ylabel("Average Marks", fontsize=12)
        plt.ylim(0, max_mark + 0.5)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{height:.2f}',
                     ha='center', va='bottom', fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "average_grades_comparison.png"), dpi=300)
        plt.close()

        # Plot 5: Grading Accuracy Pie Chart
        plt.figure(figsize=(8, 8))
        diff = df["llm_grade"] - df["teacher_marks"]
        exact_match = sum(diff == 0)
        ai_higher = sum(diff > 0)
        ai_lower = sum(diff < 0)
        
        categories = []
        counts = []
        colors = []
        
        if exact_match > 0:
            categories.append(f"AI Matches Teacher ({exact_match})")
            counts.append(exact_match)
            colors.append('#2ecc71')
        if ai_higher > 0:
            categories.append(f"AI Graded Higher ({ai_higher})")
            counts.append(ai_higher)
            colors.append('#e74c3c')
        if ai_lower > 0:
            categories.append(f"AI Graded Lower ({ai_lower})")
            counts.append(ai_lower)
            colors.append('#f1c40f')
            
        plt.pie(
            counts, 
            labels=categories, 
            colors=colors, 
            autopct='%1.1f%%', 
            startangle=140, 
            textprops={'fontsize': 11, 'weight': 'bold'},
            wedgeprops={'edgecolor': 'white', 'linewidth': 2, 'antialiased': True}
        )
        plt.title("How well does the AI agree with the Teacher?", fontsize=14, fontweight="bold", pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "grading_accuracy_pie.png"), dpi=300)
        plt.close()
        
        logger.info(f"All validation plots successfully saved to: {plots_dir}")
    else:
        logger.info("No ground truth teacher marks available. Generating basic grade distribution plot...")
        sns.set_theme(style="whitegrid")
        
        # Basic Grade Distribution Histogram
        plt.figure(figsize=(8, 6))
        sns.histplot(df["llm_grade"], kde=True, color="#2ecc71", bins=10)
        plt.title("Distribution of AI Assigned Grades", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Predicted Grade", fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "ai_grade_distribution_hist.png"), dpi=300)
        plt.close()
        logger.info(f"Basic grade distribution plot saved to: {plots_dir}")
