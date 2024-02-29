---
gpu_platforms:
  - name: ARC3
    hardware: 2 nodes with 2 x NVIDIA K80s, 6 nodes with 4 x NVIDIA P100s
    access:
        text: Anyone at Leeds
        url: https://arcdocs.leeds.ac.uk/getting_started/request_hpc_acct.html
    project: ARC4
  - name: JASMIN
    organization: NERC
    hardware: 6x NVidia GV100GL across 3 nodes
    access:
        url: https://www.jasmin.ac.uk
  - name: DiRAC (Tursa)
    organization: EPCC
    hardware: 144 nodes with 4x Nvidia RedStone A100-40 each
    access:
        text: Annual call for project proposals
        url: https://dirac.ac.uk/community/#AccessingDiRAC
    additional_info: https://www.epcc.ed.ac.uk/hpc-services/dirac-tursa-gpu
  - name: MAGEO
    organization: PML
    hardware: 5 NVIDIA DGX-1 MaxQ nodes
    access:
        text: NERC related research
        url: Via email (see website)
    additional_info: https://mageohub.neodaas.ac.uk/
  - name: BEDE
    organization: Durham
    hardware: 38x Nvidia V100 across 38 nodes in cluster
    access:
        text: EPSRC funded projects (pre EPSRC-application access available)
    additional_info: 
        - https://n8cir.org.uk/supporting-research/facilities/bede/docs/bede_registrations/
        - https://bede-documentation.readthedocs.io/en/latest/usage/index.html
  - name: JADE II
    organization: Oxford/ATI
    hardware: 63 DGX MAX-Q nodes with 8x NVidia V100 each
  - name: LEARN
    organization: UoL
    hardware: 1 DGX A100
  - name: Google Colab
    organization: Google
    hardware: NVidia k80s
    access:
        text: Free through Google account
        url: https://cloud.google.com/gpu/
    additional_info: https://www.tutorialspoint.com/google_colab/google_colab_using_free_gpu.htm
  - name: Kaggle Notebooks
    organization: Kaggle
    hardware: 30 GPU hrs/week
    access:
        text: Free through Kaggle account
        url: https://www.kaggle.com/code/dansbecker/running-kaggle-kernels-with-a-gpu
    additional_info: https://www.kaggle.com/product-feedback/173129
  - name: Gradient
    organization: Paperspace
    hardware: M4000 in free tier
    access:
        text: Free or paid tiers
  - name: Graphcore
    organization: Paperspace
    hardware: Graphcore IPU-POD16
    access:
        max_hours_per_day: 6hrs/day
        text: Free during evaluation period
    additional_info: https://www.paperspace.com/graphcore/
---

Below is an overview of GPU platforms available for research by staff at the University of Leeds:

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
