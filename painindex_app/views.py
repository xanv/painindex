from django.shortcuts import get_object_or_404, render

from painindex_app.models import PainSource

# Create your views here.
# this is the controller

def homepage(request):
    return render(request, 'painindex_app/homepage.html')

def painsource_detail(request, painsource_id):
    painsource = get_object_or_404(PainSource, pk=painsource_id)
    return render(request, 'painindex_app/painsource_detail.html', {'painsource': painsource})

def painreport_form(request):
    return render(request, 'painindex_app/painreport.html')
