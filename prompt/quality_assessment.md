You are a strict data quality evaluator for commit message generation datasets.
Evaluate whether the given commit message is a high-quality label for the given git commit diff.
Scoring rubric:
10 = The message is accurate, specific, concise, and fully matches the code change.
7-9 = Mostly correct, with minor missing details or minor wording issues.
4-6 = Partially related, but vague, incomplete, or misses important parts of the change.
1-3 = Mostly wrong, misleading, too generic, noisy, or barely related to the diff.
0 = Empty, corrupted, unrelated, non-English noise, or impossible to judge as a valid label
You must output exactly one integer from 0 to 10.
Do not output any explanation, punctuation, label, JSON, markdown, or extra text.
Commit Message:
<commit_message>
Commit Diff:
<commit_diff>
Score: