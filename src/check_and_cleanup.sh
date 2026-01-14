# Radonweave (https://github.com/Jackksonns/radonweave): low-dose CT reconstruction/denoising toolkit.
cd /root/autodl-tmp/datasets/lodopab/

cat > check_and_cleanup.sh << 'EOF'
#!/bin/bash

zip_files=(
  "observation_train.zip"
  "observation_validation.zip"
  "observation_test.zip"
  "ground_truth_train.zip"
  "ground_truth_validation.zip"
  "ground_truth_test.zip"
)

for zip in "${zip_files[@]}"; do
  if [ ! -f "$zip" ]; then
    echo "$zip does not exist, skipping"
    continue
  fi

  # Get ZIP filename prefix
  prefix="${zip%.zip}"

  # Check the list of files inside the ZIP
  zip_contents=$(unzip -Z1 "$zip" 2>/dev/null)
  if [ -z "$zip_contents" ]; then
    echo "$zip cannot be read, it may be corrupted"
    continue
  fi

  # Check line by line whether each file already exists in the current directory
  all_exist=true
  while IFS= read -r file; do
    if [ ! -f "$file" ]; then
      all_exist=false
      break
    fi
  done <<< "$zip_contents"

  if [ "$all_exist" = true ]; then
    echo "$zip has been fully extracted"
    # If you want to delete it automatically, uncomment the next line
    # rm "$zip" && echo "Deleted $zip"
  else
    echo "$zip has not been fully extracted yet (or some files are missing)"
  fi
done
EOF

chmod +x check_and_cleanup.sh
