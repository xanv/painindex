import random
from django.shortcuts import get_object_or_404, render
from painindex_app.models import PainSource


def homepage(request):
    find_bugs = [PainSource.objects.select_random_in_range(i - 0.5, i + 0.5) for i in range(10,0,-1)]
    bugs = filter(None, find_bugs)
    highlighted_bug = random.choice(bugs)

    content = {"find_bugs": find_bugs, "highlighted_bug": highlighted_bug}
    return render(request, 'painindex_app/homepage.html', content)

def painsource_detail(request, painsource_id):
    painsource = get_object_or_404(PainSource, pk=painsource_id)
    return render(request, 'painindex_app/painsource_detail.html', {'painsource': painsource})

def painreport_form(request):
    return render(request, 'painindex_app/painreport.html')
