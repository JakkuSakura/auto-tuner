Focus on one coding-quality aspect: prefer direct, explicit, readable attribute access (e.g. `obj.value`) over
reflection/introspection patterns (e.g. `getattr(obj, "value")`, `hasattr`, `vars`, `__dict__`).

When selection is required, use an explicit mapping/registry or typed protocol instead of dynamic lookup.

Do not include or mention any specific code snippet from the user’s codebase.

Hard rule: do not use Python reflection/introspection-based access patterns such as `getattr(`, `hasattr(`, `.__dict__`,
or `vars(` in the output.
