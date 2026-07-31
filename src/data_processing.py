import pandas as pd
import numpy as np


def clean_data(df_orig):
    """
    Processes data and prints statistics at each processing step.
    :param df_orig: The original DataFrame to be processed.
    :return: Cleaned DataFrame.
    """
    print(f'Responses: {len(df_orig)}')
    print(f'Domains: {df_orig["domain_id"].nunique()}')
    print(f'Tasks: {df_orig["task_id"].nunique()}')
    print(f'Users: {df_orig["user_id"].nunique()}')
    print('----------------------------')

    # Remove NULL answers
    df_null_answers = df_orig[df_orig['answer'] == 'null']
    
    # Remove time-outs
    df_timeouts = df_orig[df_orig['timed_out']]
    
    # Check for empty answers and remove
    df_timeouts_no_response = df_timeouts[df_timeouts['answer'] == 'null']
    
    # Missing social condition group
    df_no_condition = df_orig[df_orig['experimental_condition'].isna()]
    
    df = df_orig[(df_orig['answer'] != 'null') & (~df_orig['experimental_condition'].isna())]

    print(f'Responses NULL: {len(df_null_answers)} ({round(len(df_null_answers)/len(df_orig)*100, 0)}%)')
    print(f'Responses timed-out without response: {len(df_timeouts_no_response)} ({round(len(df_timeouts_no_response)/len(df_orig)*100, 0)}%)')
    print(f'Responses without an experimental condition: {len(df_no_condition)} ({round(len(df_no_condition)/len(df_orig)*100, 0)}%)')
    print(f'Total nr responses removed: {len(df_orig) - len(df)}')
    print(f'Responses remaining: {len(df)}')
    print(f'Number of users remaining: {df["user_id"].nunique()}')
    print('----------------------------')

    # Users that didn't complete domains
    quitters = df_orig.groupby(['user_id', 'domain_id']).size().reset_index(name='count')
    quitters = quitters[quitters['count'] < 20]

    # Responses without confidence
    no_confidence = df[df['confidence'] == 0]
    
    print(f'Responses without confidence: {len(no_confidence)} ({round(len(no_confidence)/len(df)*100, 0)}%)')
    print(f'Responses timed-out: {len(df_timeouts)} ({round(len(df_timeouts)/len(df)*100, 0)}%)')
    print(f'Incomplete hits: {len(quitters)} ({round(len(quitters)/len(df)*100, 0)}%)')
    print('----------------------------')
    
    # Rearrange columns
    cols = ['user_id', 'task_id', 'domain_id', 'domain_name', 'domain_description', 'prompt', 'knowledge_type', 'media_type', 
            'answer_type', 'time_limit', 'start_time', 'end_time', 'time_spent_on_question', 'timed_out', 'possible_answers', 
            'experimental_condition', 'cues', 'answer', 'correct_answer', 'is_correct', 'confidence', 'age', 'gender', 
            'education', 'industry', 'predicted_rank']
    
    df = df[cols]
    return df


# Load datasets
domains = pd.read_csv('../experiment_data/original/domains.csv.zip')
tasks = pd.read_csv('../experiment_data/original/tasks.csv.zip')
answers = pd.read_csv('../experiment_data/original/answers.csv.zip')
users = pd.read_csv('../experiment_data/original/users.csv')
ranks = pd.read_csv('../experiment_data/original/predicted_ranks.csv')

# Merge datasets
df_crowd = answers.merge(tasks, on='task_id', how='left')
df_crowd = df_crowd.merge(domains, on='domain_id', how='left')
df_crowd = df_crowd.merge(users, on='user_id', how='left')
df_crowd = df_crowd.merge(ranks, on=['domain_id', 'user_id'], how='left')

# Clean and process data
df_crowd['start_time'] = pd.to_datetime(df_crowd['start_time'])
df_crowd['end_time'] = pd.to_datetime(df_crowd['end_time'])
df_crowd['time_spent_on_question'] = (df_crowd['end_time'] - df_crowd['start_time']).dt.total_seconds()
df_crowd['timed_out'] = df_crowd['time_spent_on_question'] > df_crowd['time_limit']

df_crowd['is_correct'] = np.where(df_crowd['answer_type'] == 'open-ended', np.nan,
                                   np.where((df_crowd['answer_type'] == 'discrete') & (df_crowd['answer'] == df_crowd['correct_answer']), True, False))

# Apply cleaning function
crowd = clean_data(df_crowd)

# Save the cleaned data
crowd.to_pickle('../experiment_data/crowd.pkl')