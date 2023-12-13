---
layout: page
title: Events
permalink: /events/
seminar_data:
  - title: Forecasting Global Weather with Graph Neural Networks
    datetime: 2023-11-10T15:00:00Z
    speaker: Ryan Keisler
    media:
      youtube: https://www.youtube.com/watch?v=-CDf4GoWU3g
  - title: Could deep learning methods replace numerical weather models?
    datetime: 2023-10-20T15:00:00Z
    speaker: Mariana Clare
    media:
      youtube: https://www.youtube.com/watch?v=5BMRkXi3GgU
  - title: Feature-Preserving Point Cloud Simplification with Gaussian Processes
    datetime: 2023-07-14T14:00:00Z
    speaker: Thomas. M. McDonald
    media:
      youtube: https://www.youtube.com/watch?v=9yPHRLRYkUo
    - title: Physics-based domain adaptation for dynamical systems forecasting
    datetime: 2023-05-19T14:00:00Z
    speaker: Zack Xuereb Conti
    media:
      youtube: https://www.youtube.com/watch?v=q-YoY_3Y74M
  - title: "ClimaX: A foundation model for weather and climate"
    datetime: 2023-03-17T15:00:00Z
    speaker: Tung Nguyen (UCLA)
    media:
      youtube: https://www.youtube.com/watch?v=0pV0K1A9RvY
  - title: Sea ice detection from concurrent visible and SAR imagery using a convolutional neural network
    datetime: 2023-02-24T11:00:00Z
    speaker: Martin Rogers (British Antarctic Survey)
    media:
      youtube: https://www.youtube.com/watch?v=U4amljFGkiw
  - title: Physics-informed Machine Learning for Trustworthy Climate Emulators
    datetime: 2023-02-10T14:00:00Z
    speaker: Björn Lütjens (MIT)
    media:
      youtube: https://www.youtube.com/watch?v=826wF2DNTms
  - title: Explainable AI for identifying regional climate change patterns
    datetime: 2023-01-13T14:00:00Z
    speaker: Zack Labe (Princeton)
    media:
      youtube: https://www.youtube.com/watch?v=ZIZbu1o33xQ&t=4s
  - title: Extending the capabilities of data-driven reduced-order models to make predictions for unseen scenarios
    datetime: 2022-11-18T11:00:00Z
    speaker: Claire Heaney (Imperial College)
    media:
      youtube: https://www.youtube.com/watch?v=d1QXWSDgmEg
workshop_data:
  - title: Physics-informed neural networks (PINNs)
    datetime: 2023-02-17T14:00:00Z
    speaker: Fergus Shone (University of Leeds)
    media:
      youtube: https://www.youtube.com/watch?v=zYi8KO4rLwg&t=139s
  - title: Reinforcement Learning
    datetime: 2023-04-17T14:00:00Z
    speaker: Alhanof Alolyan (University of Leeds)
    media:
      notebook: https://sciml-leeds.github.io/categories/#reinforcement_learning
  - title: SINDy
    datetime: 2023-10-27T14:00:00Z
    speaker: Alasdair Roy (University of Leeds)
    media:
      youtube: https://www.youtube.com/watch?v=jTDym45joAk
      notebook: https://sciml-leeds.github.io/workshop/sindy/2023/10/27/SINDy_Challenges.html
---

Below is an overview of events organised by the SciML community. We aim to put
recordings of each event online within a week of them taking place.

[Please get in touch <i class="fa fa-envelope"></i>](mailto://{{site.contact_email}}) if you'd like to come to
talk to us, we're a very diverse group in terms of applications in scientific
machine learning and would <i class="fa fa-heart"></i> to hear about your work!

<h2>Seminars</h2>

<table>
<tr>
<th>Title</th>
<th>Speaker</th>
<th>Date</th>
<th>Media</th>
</tr>

{% for talk in page.seminar_data %}
<tr>
<td>{{talk.title}}</td>
<td>{{talk.speaker}}</td>
<td>{{talk.datetime | date: "%d/%m/%Y %H:%M"}}</td>
<td>
{% if talk.media.youtube %}
<a href="{{talk.media.youtube}}">recording</a>
{% endif %}
{% if talk.media.notebook %}
<a href="{{talk.media.notebook}}">code</a>
{% endif %}
</td>
</tr>
{% endfor %}

</table>

<h2>Workshops</h2>

<table>
<tr>
<th>Title</th>
<th>Speaker</th>
<th>Date</th>
<th>Media</th>
</tr>

{% for talk in page.workshop_data %}
<tr>
<td>{{talk.title}}</td>
<td>{{talk.speaker}}</td>
<td>{{talk.datetime | date: "%d/%m/%Y %H:%M"}}</td>
<td>
{% if talk.media.youtube %}
<a href="{{talk.media.youtube}}">recording</a>
{% endif %}
{% if talk.media.notebook %}
<a href="{{talk.media.notebook}}">code</a>
{% endif %}
</td>
</tr>
{% endfor %}

</table>
