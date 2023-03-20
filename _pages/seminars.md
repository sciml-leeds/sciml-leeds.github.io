---
layout: page
title: Seminars
permalink: /seminars/
data:
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
---

Below is an overview seminars organised by the SciML community. We aim to put
recordings of each seminar online within a week of the seminar taking place.

[Please get in touch <i class="fa fa-envelope"></i>](mailto://{{site.contact_email}}) if you'd like to come to
talk to us, we're a very diverse group in terms of applications in scientific
machine learning and would <i class="fa fa-heart"></i> to hear about your work!

<table>
<tr>
<th>Title</th>
<th>Speaker</th>
<th>Date</th>
<th>Media</th>
</tr>

{% for talk in page.data %}
<tr>
<td>{{talk.title}}</td>
<td>{{talk.speaker}}</td>
<td>{{talk.datetime | date: "%d/%m/%Y %H:%M"}}</td>
<td>
{% if talk.media.youtube %}
<a href="{{talk.media.youtube}}">recording</a>
{% endif %}
</td>
</tr>
{% endfor %}

</table>

