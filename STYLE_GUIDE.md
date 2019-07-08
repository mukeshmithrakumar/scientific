#### Python
Python code should conform to [PEP8](https://www.python.org/dev/peps/pep-0008/).

TF Scientific uses [flake8](http://flake8.pycqa.org/en/latest/) to format code.

You can disable them locally like this:

```python
# To ignore all errors for an entire file use:
# flake8: noqa
from foo import unused
function_that_doesnt_exist()

# To ignore a particular error for a particular line:
example = lambda: 'example'  # noqa: E731
```

#### TensorFlow Conventions

Follow the guidance in the [TensorFlow Style Guide - Conventions](https://www.tensorflow.org/community/contribute/code_style#tensorflow_conventions_and_special_uses).
