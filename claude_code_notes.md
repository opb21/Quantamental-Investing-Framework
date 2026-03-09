## Claude Code Tips

- Using `#` in a prompt tells Claude to store something to memory
- you can mention a specific file using `@`
- plan mode: shift + tab tab, gets Claude to plan a step and show you the idea for approval before implementing
- you can instruct to stage and commit changes and it will do so and write a commit message for you
- `Escape` stops Claude in its tracks. A powerful way to make Claude more accurate is to combine `Escape` with `#`
- Hit `Escape` `Escape` to go back to a prior stage of the conversation- this can be useful when you've gone on a tangent and solved an issue that is no longer relevant to the main convo.
- `/compact` will get Claude to compact what its learned so far to free up space but retain knowledge
- `/clear` will get Claude to clear memory so you can start fresh
- Hooks are actions that can be taken before or after Claude responds to your commands. There are pre-execution and post-execution hooks. 
  - An example of a pre-execution hook might be 'check if something similar to this function already exists in a folder and if it does use and update that rather than writing a new function.'
  - A post execution hook might be, 'following the changes made to this function, check whether that now breaks any dependencies where the old version of the function is used elsewhere, and if so update those function calls'
- The Claude SDK allows you to call Claude within a hook or instruction- so you instruct Claude to do something, and if it references the file with the Claude SDK that will launch another version of Claude that it can interact with