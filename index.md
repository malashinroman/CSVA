---
layout: default
---
### Briefly:
CSVA is a robust and fast algorithm for outlier elimination of keypoint matches. It can be used by developers and researches in different practical tasks with minor adaptation. 


# CSVA

"Core" structural verification algorithm (CSVA) is outlier elimination algorithm that can be used by developers and researches in different practical tasks with minor adaptation.It is proposed the sever conditions for a variety of applications and feature extraction methods. The main idea is to cluster features whose parameters agree in similarity transform space. You can think about the algorithm as (and in fact it is) the improved and generalized version of Hough Clustering described in SIFT paper by D.Lowe. 

The c++ implementation is opesource

## Outlier elimination in keypoints-based methods

Outlier elimination is a crucial stage in keypoints-based methods (SIFT, SURF, BRISK) especially in extreme conditions, for example when matching indoor scenes under severe viewpoint and lightning changes or when matching of aerial and cosmic photographs with strong appearance changes caused by season, day-time and viewpoint variation. 

CSVA referes to the IV stage of the next figure

![Feature based methods](https://raw.githubusercontent.com/malashinroman/CSVA/gh-pages/_site/assets/matching_keypoints.jpg)


## Algorithm pipeline

The proposed algorithm pipeline involves:
1. Many-to-one matches exclusion, 
2. The improved Hough Clustering of keypoint matches
3. Cluster verification procedure based on modified RANSAC. 

![Pipeline](https://raw.githubusercontent.com/malashinroman/CSVA/gh-pages/assets/pipeline.png) <!-- .element height="20%" width="20%" -->


--[An image](images/an_image.jpg) 
### Example results

Registering aerospace images under season changes


![Registering aerospace images under season changes](https://raw.githubusercontent.com/malashinroman/CSVA/gh-pages/_site/assets/season_change.png)

Registering old and new images of NewYork: 1925 vs 2014

![Registering old and new images of NewYork: 1925 vs 2014](https://raw.githubusercontent.com/malashinroman/CSVA/gh-pages/_site/assets/New_York_result.png)

Filtered keypoint matches refer to the stable bridge configuration

![Filtered keypoint matches](https://raw.githubusercontent.com/malashinroman/CSVA/gh-pages/_site/assets/New_York_result.png)

![Registering old maps of Moscow](https://raw.githubusercontent.com/malashinroman/CSVA/gh-pages/_site/assets/Map_of_Moscow.png)

![Matching under strong appearence change](https://raw.githubusercontent.com/malashinroman/CSVA/gh-pages/_site/assets/Peter_the_Great.jpg)


> This is a blockquote following a header.
>
> When something is important enough, you do it even if the odds are not in your favor.
[Link to another page](./another-page.html).

There should be whitespace between paragraphs.

There should be whitespace between paragraphs. We recommend including a README, or a file with information about your project.


Text can be **bold**, _italic_, or ~~strikethrough~~.

```c++
// Javascript code with syntax highlighting.
var fun = function lang(l) {
  dateformat.i18n = require('./lang/' + l)
  return true;
}
```

```ruby
# Ruby code with syntax highlighting
GitHubPages::Dependencies.gems.each do |gem, version|
  s.add_dependency(gem, "= #{version}")
end
```

#### Header 4

*   This is an unordered list following a header.
*   This is an unordered list following a header.
*   This is an unordered list following a header.

##### Header 5

1.  This is an ordered list following a header.
2.  This is an ordered list following a header.
3.  This is an ordered list following a header.

###### Header 6

| head1        | head two          | three |
|:-------------|:------------------|:------|
| ok           | good swedish fish | nice  |
| out of stock | good and plenty   | nice  |
| ok           | good `oreos`      | hmm   |
| ok           | good `zoute` drop | yumm  |

### There's a horizontal rule below this.

* * *

### Here is an unordered list:

*   Item foo
*   Item bar
*   Item baz
*   Item zip

### And an ordered list:

1.  Item one
1.  Item two
1.  Item three
1.  Item four

### And a nested list:

- level 1 item
  - level 2 item
  - level 2 item
    - level 3 item
    - level 3 item
- level 1 item
  - level 2 item
  - level 2 item
  - level 2 item
- level 1 item
  - level 2 item
  - level 2 item
- level 1 item

### Small image

![Octocat](https://assets-cdn.github.com/images/icons/emoji/octocat.png)

### Large image

![Branching](https://guides.github.com/activities/hello-world/branching.png)


### Definition lists can be used with HTML syntax.

<dl>
<dt>Name</dt>
<dd>Godzilla</dd>
<dt>Born</dt>
<dd>1952</dd>
<dt>Birthplace</dt>
<dd>Japan</dd>
<dt>Color</dt>
<dd>Green</dd>
</dl>

```
Long, single-line code blocks should not wrap. They should horizontally scroll if they are too long. This line should be long enough to demonstrate this.
```

```
The final element.
```
