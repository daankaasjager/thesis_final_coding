# missing from TOML:
comm -23 <(sort requirements.txt) <(sort /tmp/poetry_reqs.txt) \
  | xargs -r -n1 echo "poetry add"

# extras in TOML (you might remove these if they really aren’t used):
comm -13 <(sort requirements.txt) <(sort /tmp/poetry_reqs.txt) \
  | xargs -r -n1 echo "poetry remove"
