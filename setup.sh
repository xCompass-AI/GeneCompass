PROJECT_DIR=$(pwd)

# Check if the directory is already in PYTHONPATH
if [[ ":$PYTHONPATH:" == *":$PROJECT_DIR:"* ]]; then
  echo "Project directory is already in PYTHONPATH."
else
  # Add the directory to PYTHONPATH
  echo "export PYTHONPATH=\"$PROJECT_DIR:\$PYTHONPATH\"" >> ~/.bashrc  # For bash users
  echo "export PYTHONPATH=\"$PROJECT_DIR:\$PYTHONPATH\"" >> ~/.zshrc   # For zsh users (if using zsh)

  # Apply the changes immediately to the current shell
  export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

  echo "Project directory added to PYTHONPATH."
fi

# pip install -r requirements.txt