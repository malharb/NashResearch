import pandas as pd
import os

# Step 1: Get the first 100 unique person_ids who have depression and all Fitbit data
unique_persons_sql = """
WITH latest_condition_date AS (
  SELECT
    person_id,
    CAST(MAX(condition_start_datetime) AS DATE) AS latest_condition_start_date
  FROM
    `""" + os.environ["WORKSPACE_CDR"] + """.condition_occurrence`
  WHERE
    condition_concept_id = 4282096
  GROUP BY
    person_id
),
eligible_persons AS (
  SELECT
    DISTINCT person_id
  FROM
    `""" + os.environ["WORKSPACE_CDR"] + """.cb_search_person`
  WHERE
    person_id IN (
      SELECT
        person_id
      FROM
        `""" + os.environ["WORKSPACE_CDR"] + """.cb_search_person`
      WHERE
        has_fitbit_heart_rate_level = 1
        AND has_fitbit_steps_intraday = 1
        AND has_fitbit_sleep_level = 1
        AND has_fitbit_activity_summary = 1
    )
)
SELECT
  ep.person_id,
  lcd.latest_condition_start_date
FROM
  eligible_persons ep
JOIN
  latest_condition_date lcd
ON
  ep.person_id = lcd.person_id
WHERE
  ep.person_id IN (
    SELECT
      person_id
    FROM
      `""" + os.environ["WORKSPACE_CDR"] + """.condition_occurrence`
    WHERE
      condition_concept_id = 4282096
  )
LIMIT 5
"""

# Execute the query to get the first 100 unique person_ids
unique_persons_df = pd.read_gbq(
    unique_persons_sql,
    dialect="standard",
    use_bqstorage_api=("BIGQUERY_STORAGE_API_ENABLED" in os.environ),
    progress_bar_type="tqdm_notebook"
)

# Extract the list of unique person_ids and their latest condition start date
unique_person_ids = unique_persons_df['person_id'].tolist()
latest_condition_dates = unique_persons_df.set_index('person_id')['latest_condition_start_date'].to_dict()

# Create the subquery for each person_id
subqueries = []
for person_id in unique_person_ids:
    latest_condition_start_date = latest_condition_dates[person_id]
    subquery = f"""
    SELECT
      person_id,
      datetime AS heart_rate_datetime,
      heart_rate_value
    FROM
      `{os.environ["WORKSPACE_CDR"]}.heart_rate_minute_level`
    WHERE
      person_id = {person_id}
      AND CAST(datetime AS DATE) <= '{latest_condition_start_date}'
    """
    subqueries.append(subquery)

# Combine all subqueries using UNION ALL
combined_query = "\nUNION ALL\n".join(subqueries)

# Execute the combined query to get the heart rate data
all_heart_rate_data = pd.read_gbq(
    combined_query,
    dialect="standard",
    use_bqstorage_api=("BIGQUERY_STORAGE_API_ENABLED" in os.environ),
    progress_bar_type="tqdm_notebook"
)

# Display the first 5 rows of the combined heart rate data
print(all_heart_rate_data.head(5))

# Check the number of unique person_ids in the result
unique_person_ids_count = all_heart_rate_data['person_id'].nunique()
print(f'Number of unique person_ids in the result: {unique_person_ids_count}')
