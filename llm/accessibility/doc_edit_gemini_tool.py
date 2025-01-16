"""
Reqs:
1. Brings in sections of document, knowing which it's bringing in
2. Reads them to the editor (gemini allows interruptions for edits)
3. (For now) writes the edit (after some verbal confirmation) to the remote file via a function provided to call

Functions:
1. bring in the right section for review: let's assume I have a json with each section parsed (let's assume abstract, intro, methods, results, discussion), so just section_text = f(section_name)
2. submit edits to be written to the stored document. confirmed = f(prior_text, new_text) --> opportunity here for fuzzy matching because edits can be more complex, or even LLM interpretation of what the change should be

MVP:
1. locally serves a json file (actually let's just put it into gemini for now)
2. Bottleneck is function calling via gemini

"""