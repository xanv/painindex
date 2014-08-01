from django.conf.urls import patterns, url
from django.views.generic import TemplateView, FormView



urlpatterns = patterns('',
    url(r'^$', TemplateView.as_view(template_name="painindex_app/homepage.html"), name='homepage'),
    
            
)