from django.shortcuts import get_object_or_404, render

from painindex_app.models import PainSource

# Create your views here.
# this is the controller

def homepage(request):
    # this is irritatingly complicated because our database is small.  it will be much simpler once there is at least one bug with an average rating of approximately n for in in 1 to 10.
    import random
    find_bugs = []
    bugs = []

    for num in reversed(range(1, 11)):
        find_bugs.append(PainSource.objects.in_range(num - 0.5, num + 0.5))

    for potential_bug in find_bugs:
        if potential_bug == None:
            continue
        else:
            bugs.append(potential_bug)


    rand_bug_num = random.randint(0, len(bugs) - 1)
    highlighted_bug = bugs[rand_bug_num]

    content = {"find_bugs": find_bugs, "bugs": bugs, "highlighted_bug": highlighted_bug}
    print content
    return render(request, 'painindex_app/homepage.html', content)

def painsource_detail(request, painsource_id):
    painsource = get_object_or_404(PainSource, pk=painsource_id)
    return render(request, 'painindex_app/painsource_detail.html', {'painsource': painsource})

def painreport_form(request):
    return render(request, 'painindex_app/painreport.html')
