import os
import sys
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Make sure evaluator is importable from current directory
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from evaluator import evaluate_single

# Create required directory structure
os.makedirs("datasets", exist_ok=True)
os.makedirs("output", exist_ok=True)
os.makedirs(os.path.join("output", "plots"), exist_ok=True)
os.makedirs("results_backup", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# -----------------------------
# CONFIGURATION
# -----------------------------
START_IDX = 0
END_IDX = 4274
OUTPUT_DIR = os.path.join("output", "plots")
RESULTS_DIR = "results_backup"
DATASET_PATH = os.path.join(
    "datasets",
    "train.csv"
)

# -----------------------------
# STEP 2: RUN EVALUATIONS
# -----------------------------
def run_evaluation(csv_path):
    global END_IDX
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Dataset loaded. Total rows: {len(df)}")
    
    if END_IDX is None:
        END_IDX = len(df)
        
    print(f"Processing rows {START_IDX} to {END_IDX}")

    sample_df = df.iloc[
        START_IDX:END_IDX
    ].copy()
    
    # Checkpoint / Resume logic
    latest_results_path = os.path.join(RESULTS_DIR, "latest_results.csv")
    results = []
    completed_rows = 0
    if os.path.exists(latest_results_path):
        try:
            latest_df = pd.read_csv(latest_results_path)
            completed_rows = len(latest_df)
            if completed_rows > 0:
                results = latest_df.to_dict(orient="records")
                print(f"Resuming from row {START_IDX + completed_rows}")
                print(f"Continue from: {START_IDX + completed_rows + 1}")
        except Exception as e:
            print(f"Error reading checkpoint: {e}")
            completed_rows = 0
            
    # If we have already processed all rows in the sample, return immediately
    if completed_rows >= len(sample_df):
        print("All rows in sample have already been processed.")
        return pd.DataFrame(results)
        
    print(f"\nStarting evaluation of {len(sample_df)} answers...")
    
    backoffs = [20, 40, 60, 80, 100]
    
    # Loop over remaining rows
    for idx, (_, row) in enumerate(sample_df.iloc[completed_rows:].iterrows(), completed_rows + 1):
        q = row.get("questions", "")
        student_ans = row.get("student_answer", "")
        model_ans = row.get("model_answer", "")
        teacher_marks = float(row.get("teacher_marks", 0))
        total_marks = float(row.get("total_marks", 5))
        
        print(f"\n[{idx}/{len(sample_df)}] Question: {q[:50]}...")
        print(f"  - Student: {str(student_ans)[:50]}...")
        print(f"  - Teacher Marks: {teacher_marks} / {total_marks}")
        
        start_time = time.time()
        success = False
        eval_res = None
        retry_count = 0
        
        # Retry logic
        for attempt in range(6): # 0 to 5
            try:
                eval_res = evaluate_single(q, model_ans, student_ans, max_marks=total_marks)
                success = True
                break
            except Exception as e:
                if "429" in str(e):
                    if attempt < 5:
                        delay = backoffs[attempt]
                        retry_count = attempt + 1
                        print(f"Rate limit hit. Waiting {delay}s...")
                        time.sleep(delay)
                    else:
                        print(f"  [ERROR] Max retries reached for question {idx}. Skipping.")
                else:
                    print(f"  [ERROR] Failed to grade due to non-429 error: {e}")
                    break
        
        elapsed = time.time() - start_time
        
        # Log to logs/run.log
        q_clean_log = str(q).replace("\n", " ").replace("\r", " ")
        try:
            with open(os.path.join("logs", "run.log"), "a", encoding="utf-8") as log_file:
                if success and eval_res is not None:
                    log_file.write(f"Index: {idx} | Question: {q_clean_log} | Grade: {eval_res['llm_grade']} | Elapsed: {elapsed:.2f}s | Retries: {retry_count}\n")
                else:
                    log_file.write(f"Index: {idx} | Question: {q_clean_log} | Grade: SKIPPED | Elapsed: {elapsed:.2f}s | Retries: {retry_count}\n")
        except Exception as log_err:
            print(f"Failed to write to log: {log_err}")
            
        if not success or eval_res is None:
            # Polite sleep between answers
            time.sleep(5)
            continue
            
        print(f"  - LLM Assigned: {eval_res['llm_grade']} (in {elapsed:.2f}s)")
        
        results.append({
            "question": q,
            "model_answer": model_ans,
            "student_answer": student_ans,
            "teacher_marks": teacher_marks,
            "total_marks": total_marks,
            "llm_grade": eval_res["llm_grade"],
            "sbert_score": eval_res["sbert_score"],
            "bert_score": eval_res["bert_score"],
            "feedback": eval_res["llm_feedback"]
        })
        
        # Checkpointing every 20 answers
        if idx % 20 == 0:
            try:
                checkpoint_df = pd.DataFrame(results)
                checkpoint_df.to_csv(latest_results_path, index=False)
                
                checkpoint_filename = f"checkpoint_{START_IDX}_{END_IDX}_{idx}.csv"
                checkpoint_filepath = os.path.join(RESULTS_DIR, checkpoint_filename)
                checkpoint_df.to_csv(checkpoint_filepath, index=False)
                print(f"Checkpoint saved -> {checkpoint_filename}")
            except Exception as e:
                print(f"Failed to save checkpoint at row {idx}: {e}")
                
        # Polite sleep to respect rate limits (5 seconds between answers)
        time.sleep(5)
        
    return pd.DataFrame(results)

# -----------------------------
# STEP 3: PLOT GENERATION & METRICS
# -----------------------------
def analyze_and_plot(res_df):
    if res_df.empty:
        print("No evaluation data to analyze.")
        return
    
    # Save raw results
    results_filename = f"evaluation_results_{START_IDX}_{END_IDX}.csv"
    res_df.to_csv(os.path.join("output", results_filename), index=False)
    
    # Calculate errors
    mae = mean_absolute_error(res_df["teacher_marks"], res_df["llm_grade"])
    mse = mean_squared_error(res_df["teacher_marks"], res_df["llm_grade"])
    rmse = np.sqrt(mse)
    
    # Calculate correlations
    p_corr, _ = pearsonr(res_df["teacher_marks"], res_df["llm_grade"])
    s_corr, _ = spearmanr(res_df["teacher_marks"], res_df["llm_grade"])
    
    sbert_corr, _ = pearsonr(res_df["teacher_marks"], res_df["sbert_score"])
    bert_corr, _ = pearsonr(res_df["teacher_marks"], res_df["bert_score"])
    
    print("\n" + "="*50)
    print("                EVALUATION METRICS                ")
    print("="*50)
    print(f"Mean Absolute Error (MAE):     {mae:.4f}")
    print(f"Mean Squared Error (MSE):      {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Pearson Correlation (Grade):   {p_corr:.4f}")
    print(f"Spearman Correlation (Grade):  {s_corr:.4f}")
    print(f"SBERT Similarity Correlation:  {sbert_corr:.4f}")
    print(f"BERTScore Similarity Correlation: {bert_corr:.4f}")
    print("="*50 + "\n")
    
    # Write summary text
    summary_filename = f"evaluation_summary_{START_IDX}_{END_IDX}.txt"
    with open(os.path.join("output", summary_filename), "w") as f:
        f.write("AUTOMATIC SHORT ANSWER GRADING PIPELINE EVALUATION REPORT\n")
        f.write("=========================================================\n")
        f.write(f"Sample Size Evaluated: {len(res_df)}\n")
        f.write(f"Mean Absolute Error (MAE):     {mae:.4f}\n")
        f.write(f"Mean Squared Error (MSE):      {mse:.4f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")
        f.write(f"Pearson Correlation (Grade vs Ground Truth):  {p_corr:.4f}\n")
        f.write(f"Spearman Correlation (Grade vs Ground Truth): {s_corr:.4f}\n")
        f.write(f"SBERT Score Correlation with Teacher Marks:   {sbert_corr:.4f}\n")
        f.write(f"BERT Score Correlation with Teacher Marks:    {bert_corr:.4f}\n")
        f.write("=========================================================\n")
        
    # Plot settings
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Scatter plot of Grade Correlation
    plt.figure(figsize=(8, 6))
    sns.regplot(
        x="teacher_marks", 
        y="llm_grade", 
        data=res_df, 
        color="#2c3e50", 
        scatter_kws={"s": 80, "alpha": 0.7, "color": "#16a085"},
        line_kws={"color": "#e74c3c", "linewidth": 2}
    )
    # Perfect match line
    max_mark = res_df["total_marks"].max()
    plt.plot([0, max_mark], [0, max_mark], linestyle="--", color="gray", alpha=0.5, label="Perfect Agreement")
    
    plt.title("LLM Assigned Grades vs. Teacher Marks", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Teacher Marks (Ground Truth)", fontsize=12)
    plt.ylabel("LLM Assigned Grade", fontsize=12)
    plt.xlim(-0.5, max_mark + 0.5)
    plt.ylim(-0.5, max_mark + 0.5)
    plt.text(0.1, max_mark - 0.5, f"MAE: {mae:.2f}\nPearson r: {p_corr:.2f}\nSpearman ρ: {s_corr:.2f}", 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'), fontsize=10)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "grade_correlation_scatter.png"), dpi=300)
    plt.close()
    
    # Plot 2: Overlapping Distribution / KDE plot
    plt.figure(figsize=(8, 6))
    sns.kdeplot(res_df["teacher_marks"], fill=True, color="#3498db", label="Teacher Marks (Ground Truth)", alpha=0.4, bw_adjust=0.8)
    sns.kdeplot(res_df["llm_grade"], fill=True, color="#2ecc71", label="LLM Assigned Grade", alpha=0.4, bw_adjust=0.8)
    
    plt.title("Distribution of Assigned Grades", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Marks / Grade", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "grade_distribution_kde.png"), dpi=300)
    plt.close()
    
    # Plot 3: Box plot of Similarity Scores vs. Teacher Marks
    plt.figure(figsize=(10, 5))
    melted_df = res_df.melt(
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
    plt.savefig(os.path.join(OUTPUT_DIR, "similarity_scores_vs_marks.png"), dpi=300)
    plt.close()
    
    # -----------------------------
    # NEW INTUITIVE GRAPH 1: Student-by-Student Score Comparison (Grouped Bar Chart)
    # -----------------------------
    plt.figure(figsize=(10, 6))
    
    indices = np.arange(len(res_df))
    width = 0.35  # width of the bars
    
    plt.bar(indices - width/2, res_df["teacher_marks"], width, label="Teacher Marks (Ground Truth)", color="#3498db")
    plt.bar(indices + width/2, res_df["llm_grade"], width, label="AI Grade (Mistral)", color="#2ecc71")
    
    plt.title("Student-by-Student Grade Comparison", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Student Answer Samples", fontsize=12)
    plt.ylabel("Marks Assigned (out of 5)", fontsize=12)
    
    x_labels = []
    for idx, row in res_df.iterrows():
        q_text = row.get("question", "")
        # Get first 3 words of the question for the label
        words = q_text.split()
        short_q = " ".join(words[:2]) if len(words) >= 2 else q_text[:12]
        x_labels.append(f"Sample {idx+1}\n({short_q}...)")
        
    plt.xticks(indices, x_labels, fontsize=10)
    plt.ylim(0, 5.5)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "student_by_student_comparison.png"), dpi=300)
    plt.close()
    
    # -----------------------------
    # NEW INTUITIVE GRAPH 2: Overall Grading Accuracy (Pie Chart)
    # -----------------------------
    plt.figure(figsize=(8, 8))
    
    # Calculate difference: AI - Teacher
    diff = res_df["llm_grade"] - res_df["teacher_marks"]
    
    exact_match = sum(diff == 0)
    ai_higher = sum(diff > 0)
    ai_lower = sum(diff < 0)
    
    categories = []
    counts = []
    colors = []
    
    if exact_match > 0:
        categories.append(f"AI Matches Teacher Exactly ({exact_match})")
        counts.append(exact_match)
        colors.append('#2ecc71')  # Green
    if ai_higher > 0:
        categories.append(f"AI Graded Higher / Leniantly ({ai_higher})")
        counts.append(ai_higher)
        colors.append('#e74c3c')  # Red
    if ai_lower > 0:
        categories.append(f"AI Graded Lower / Harshly ({ai_lower})")
        counts.append(ai_lower)
        colors.append('#f1c40f')  # Yellow
        
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
    plt.savefig(os.path.join(OUTPUT_DIR, "grading_accuracy_pie.png"), dpi=300)
    plt.close()
    
    # -----------------------------
    # NEW INTUITIVE GRAPH 3: Average Grades (Simple Two-Bar Chart)
    # -----------------------------
    plt.figure(figsize=(6, 6))
    
    avg_teacher = res_df["teacher_marks"].mean()
    avg_ai = res_df["llm_grade"].mean()
    
    bars = plt.bar(["Teacher's Average", "AI's Average"], [avg_teacher, avg_ai], color=['#3498db', '#2ecc71'], width=0.5)
    
    plt.title("Average Score Comparison", fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("Average Marks (out of 5)", fontsize=12)
    plt.ylim(0, 5.5)
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2., 
            height + 0.1,
            f'{height:.2f}',
            ha='center', 
            va='bottom', 
            fontsize=11, 
            fontweight='bold'
        )
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "average_grades_comparison.png"), dpi=300)
    plt.close()
    
    print(f"Graphs saved successfully to folder: {OUTPUT_DIR}")
    print(f"CSV and summary saved successfully to folder: output/")

# -----------------------------
# MAIN EXECUTION
# -----------------------------
def main():
    try:
        res_df = run_evaluation(DATASET_PATH)
        analyze_and_plot(res_df)
        print("\nEvaluation successfully completed.")
    except Exception as e:
        print(f"\n[FATAL ERROR] Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
