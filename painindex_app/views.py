import random
from django.shortcuts import get_object_or_404, render
from django.views.generic.edit import CreateView
from django.core.urlresolvers import reverse_lazy
from painindex_app.models import PainSource
from painindex_app.forms import PainReportForm
from django.http import HttpResponse


def homepage(request):
    find_bugs = [PainSource.objects.select_random_in_range(i - 0.5, i + 0.5) 
        for i in range(10,0,-1)]
    # bugs = filter(None, find_bugs)
    # highlighted_bug = random.choice(bugs)

    # content = {"find_bugs": find_bugs, "highlighted_bug": highlighted_bug}
    # return render(request, 'painindex_app/homepage.html', content)
    return HttpResponse("hi")

def painsource_detail(request, painsource_id):
    painsource = get_object_or_404(PainSource, pk=painsource_id)
    return render(request, 'painindex_app/painsource_detail.html', {'painsource': painsource})

# def painreport_form(request):
#     return render(request, 'painindex_app/painreport.html')

class PainReportView(CreateView):
    form_class = PainReportForm
    template_name = 'painindex_app/painreport.html'
    # We probably want to change this:
    success_url = reverse_lazy('painindex_app:painreport')

    # This runs after form is found valid
    def form_valid(self, form):
        # Add any processing; for example, perhaps we want
        # to run calc_rating on the PainSource that's just been updated.

        return super(CreateView, self).form_valid(form)