from django.conf.urls import patterns, url
from django.views.generic import TemplateView, FormView
from painindex_app import views


urlpatterns = patterns('',
    url(r'^$', views.homepage, name='homepage'),
    url(r'^about/$', TemplateView.as_view(template_name='painindex_app/about.html'), name='about'),
    url(r'^painsource/(?P<painsource_id>\d+)/$', views.painsource_detail, name='painsource_detail'),
    # url(r'^painreport/new/$', views.painreport_form, name='painreport'),
    url(r'^painreport/new/$', views.PainReportView.as_view(), name='painreport'),    
            
)