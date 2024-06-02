# sketch/forms.py
from django import forms

class SketchForm(forms.Form):
    sketch = forms.ImageField()
