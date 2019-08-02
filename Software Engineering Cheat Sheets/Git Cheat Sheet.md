To remove a file from the staging area:

    git reset HEAD -- <file>

To remove the entire directory from the staging area:

    git reset HEAD -- <directoryName>
    
To add all changes, including deleting changes:
    
    git add -A

To delete commits. HEAD~N means to reset the N commit. Commit text editor will pop up. If you delete the text for the commit, the commit will be deleted.
    
    git rebase -i HEAD~N
Preview and then remove untracked files and directories
    git clean -nd
    git clean -fd
