from django.forms import ModelForm
from painindex_app.models import PainReport


class PainReportForm(ModelForm):
    class Meta:
        model = PainReport
        fields = ['pain_source', 'intensity', 'description']
