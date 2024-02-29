---
layout: page
title: GPU platforms
permalink: /gpu/
gpu_platforms:
  - name: Pascal (DMI)
    hardware: 1x Nvidia A100 80GB
    access:
        contact: Tore Wulf
        email: twu@dmi.dk
    project: ASIP
  - name: ATOS (ECMWF)
    hardware: 72x Nvidia A100
    access:
        url: https://confluence.ecmwf.int/display/EWCLOUDKB/GPU+support+at+ECMWF
  - name: European Weather Cloud
    hardware: 36x Nvidia A100 (TBC), imminently available (June 2023)
    access:
        text: pilot access only
  - name: LUMI (EuroHPC)
    hardware: 2560x4x AMD MI250x
    access:
        url: https://docs.lumi-supercomputer.eu/firststeps/
  - name: LEONARDO (EuroHPC)
    hardware: 3456 GPU nodes with 4 NVIDIA A100 each (Booster module), 1536 CPU nodes with 56 cores each (Data Centric module), available soon
    access:
        url: https://leonardo-supercomputer.cineca.eu/hpc-system/
---

Below is an overview of GPU platforms available for research by staff at DMI:

<h2>GPU platforms</h2>

<table>
<tr>
<th>Name</th>
<th>Hardware</th>
<th>Access</th>
<th>Project</th>
</tr>

{% for platform in page.gpu_platforms %}
<tr>
<td>{{platform.name}}</td>
<td>{{platform.hardware}}</td>
<td>
{% if platform.access.email %}
Email <a href="mailto:{{platform.contact.email}}">{{platform.access.contact}}</a>
{% elsif platform.access.url %}
<a href="{{platform.contact.url}}">{{platform.access.url}}</a>
{% elsif platform.access.text %}
{{ platform.access.text }}
{% endif %}
</td>
<td>{{platform.project}}</td>
</tr>
{% endfor %}

</table>

- [EuroHPC JU - supercomputers](https://eurohpc-ju.europa.eu/about/our-supercomputers_en)
