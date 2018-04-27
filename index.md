---
layout: default
---


# CSVA

### Briefly:
CSVA is an algorithm for outlier elimination that can be used by developers and researches in different practical tasks with minor adaptation. It was applied for matching aerospace images and images of 3D-scenes. It is much more robust and accurate than standart epipolar geometry constraint, or fitting a plane with RANSAC.


"Core" structural verification algorithm (CSVA) is outlier elimination algorithm that can be used by developers and researches in different practical tasks with minor adaptation.It is proposed the sever conditions for a variety of applications and feature extraction methods. The main idea is to cluster features whose parameters agree in similarity transform space. You can think about the algorithm as (and in fact it is) the improved and generalized version of Hough Clustering described in SIFT paper by D.Lowe. 

The c++ implementation is opesource

## Outlier elimination in keypoints-based methods

Outlier elimination is a crucial stage in keypoints-based methods (SIFT, SURF, BRISK) especially in extreme conditions, for example when matching indoor scenes under severe viewpoint and lightning changes or when matching of aerial and cosmic photographs with strong appearance changes caused by season, day-time and viewpoint variation. 

CSVA referes to the IV stage of the next figure

![Feature based methods]({{ "/assets/matching_keypoints.jpg" }})


## Algorithm pipeline

The proposed algorithm pipeline involves:
1. Many-to-one matches exclusion, 
2. The improved Hough Clustering of keypoint matches
3. Cluster verification procedure based on modified RANSAC. 

![Feature based methods]({{ "/assets/pipeline.png" }})

It is also shown that the usage of the nearest neighbour ratio may eliminate too many inliers when matching two images (especially in extreme conditions), and the preferable method is a simple many-to-one matches exclusion. The theory and experiment prove the propriety of the suggested parameters, algorithms and modifications.

### Example results

Matching aerospace images

![Feature based methods]({{ "/assets/pipeline.png" }})




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
