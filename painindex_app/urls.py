from django.conf.urls import patterns, url
from django.views.generic import TemplateView, FormView
from painindex_app import views


urlpatterns = patterns('',
    url(r'^$', views.homepage, name='homepage'),
    url(r'^painsource/(?P<painsource_id>\d+)/$', views.painsource_detail, name='painsource_detail'),
    url(r'^painreport/new/$', views.painreport_form, name='painreport'),
    
            
)