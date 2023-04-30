Download Link: https://assignmentchef.com/product/solved-cs229-problem-3-theory-unsupervised-learning
<br>



<h1>1. Uniform convergence</h1>

You are hired by CNN to help design the sampling procedure for making their electoral predictions for the next presidential election in the (fictitious) country of Elbania.

The country of Elbania is organized into states, and there are only two candidates running in this election: One from the Elbanian Democratic party, and another from the Labor Party of Elbania. The plan for making our electorial predictions is as follows: We’ll sample <em>m </em>voters from each state, and ask whether they’re voting democrat. We’ll then publish, for each state, the estimated fraction of democrat voters. In this problem, we’ll work out how many voters we need to sample in order to ensure that we get good predictions with high probability.

One reasonable goal might be to set <em>m </em>large enough that, with high probability, we obtain uniformly accurate estimates of the fraction of democrat voters in every state. But this might require surveying very many people, which would be prohibitively expensive. So, we’re instead going to demand only a slightly lower degree of accuracy.

Specifically, we’ll say that our prediction for a state is “highly inaccurate” if the estimated fraction of democrat voters differs from the actual fraction of democrat voters within that state by more than a tolerance factor <em>γ</em>. CNN knows that their viewers will tolerate some small number of states’ estimates being highly inaccurate; however, their credibility would be damaged if they reported highly inaccurate estimates for too many states. So, rather than trying to ensure that all states’ estimates are within <em>γ </em>of the true values (which would correspond to no state’s estimate being highly inaccurate), we will instead try only to ensure that the number of states with highly inaccurate estimates is small.

To formalize the problem, let there be <em>n </em>states, and let <em>m </em>voters be drawn IID from each state. Let the actual fraction of voters in state <em>i </em>that voted democrat be <em>φ<sub>i</sub></em>. Also let <em>X<sub>ij </sub></em>(1 ≤ <em>i </em>≤ <em>n,</em>1 ≤ <em>j </em>≤ <em>m</em>) be a binary random variable indicating whether the <em>j</em>-th randomly chosen voter from state <em>i </em>voted democrat:

1         if the <em>j<sup>th </sup></em>example from the <em>i<sup>th </sup></em>state voted democrat

<em>ij </em>=

0    otherwise

We assume that the voters correctly disclose their vote during the survey. Thus, for each value of <em>i</em>, we have that <em>X<sub>ij </sub></em>are drawn IID from a Bernoulli(<em>φ<sub>i</sub></em>) distribution. Moreover, the <em>X<sub>ij</sub></em>’s (for all <em>i,j</em>) are all mutually independent.

After the survey, the fraction of democrat votes in state <em>i </em>is estimated as:

Also, let <em>Z<sub>i </sub></em>= 1{|<em>φ</em>ˆ<em><sub>i </sub></em>− <em>φ<sub>i</sub></em>| <em>&gt; γ</em>} be a binary random variable that indicates whether the prediction in state <em>i </em>was highly inaccurate.

<ul>

 <li>Let <em>ψ<sub>i </sub></em>be the probability that <em>Z<sub>i </sub></em>= 1. Using the Hoeffding inequality, find an upper bound on <em>ψ<sub>i</sub></em>.</li>

 <li>In this part, we prove a general result which will be useful for this problem. Let <em>V<sub>i </sub></em>and <em>W<sub>i </sub></em>(1 ≤ <em>i </em>≤ <em>k</em>) be Bernoulli random variables, and suppose</li>

</ul>

E[<em>V<sub>i</sub></em>] = <em>P</em>(<em>V<sub>i </sub></em>= 1) ≤ <em>P</em>(<em>W<sub>i </sub></em>= 1) = E[<em>W<sub>i</sub></em>]                  ∀<em>i </em>∈ {1<em>,</em>2<em>,…k</em>}

Let the <em>V<sub>i</sub></em>’s be mutually independent, and similarly let the <em>W<sub>i</sub></em>’s also be mutually independent. Prove that, for any value of <em>t</em>, the following holds: !

[Hint: One way to do this is via induction on <em>k</em>. If you use a proof by induction, for the base case (<em>k </em>= 1), you must show that the inequality holds for <em>t &lt; </em>0, 0 ≤ <em>t &lt; </em>1, and <em>t </em>≥ 1.]

<ul>

 <li>The fraction of states on which our predictions are highly inaccurate is given by</li>

</ul>

. Prove a reasonable closed form upper bound on the probability <em>P</em>(<em>Z &gt; τ</em>) of being highly inaccurate on more than a fraction <em>τ </em>of the states.

[Note: There are many possible answers, but to be considered reasonable, your bound must decrease to zero as <em>m </em>→ ∞ (for fixed <em>n </em>and <em>τ &gt; </em>0). Also, your bound should either remain constant or decrease as <em>n </em>→ ∞ (for fixed <em>m </em>and <em>τ &gt; </em>0). It is also fine

if, for some values of <em>τ</em>, <em>m </em>and <em>n</em>, your bound just tells us that <em>P</em>(<em>Z &gt; τ</em>) ≤ 1 (the trivial bound).]

<h1>2.More VC dimension</h1>

Let the domain of the inputs for a learning problem be X = R. Consider using hypotheses of the following form:

<em>h<sub>θ</sub></em>(<em>x</em>) = 1{<em>θ</em><sub>0 </sub>+ <em>θ</em><sub>1</sub><em>x </em>+ <em>θ</em><sub>2</sub><em>x</em><sup>2 </sup>+ ··· + <em>θ<sub>d</sub>x<sup>d </sup></em>≥ 0}<em>,</em>

and let H = {<em>h<sub>θ </sub></em>: <em>θ </em>∈ R<em><sup>d</sup></em><sup>+<a href="#_ftn1" name="_ftnref1">[1]</a></sup>} be the corresponding hypothesis class. What is the VC dimension of H? Justify your answer.

[Hint: You may use the fact that a polynomial of degree <em>d </em>has at most <em>d </em>real roots. When doing this problem, you should not assume any other non-trivial result (such as that the VC dimension of linear classifiers in <em>d</em>-dimensions is <em>d </em>+ 1) that was not formally proved in class.]

<h1>3.  LOOCV and SVM</h1>

<ul>

 <li><strong>Linear Case. </strong>Consider training an SVM using a linear Kernel <em>K</em>(<em>x,z</em>) = <em>x<sup>T</sup>z </em>on a training set {(<em>x</em><sup>(<em>i</em>)</sup><em>,y</em><sup>(<em>i</em>)</sup>) : <em>i </em>= 1<em>,…,m</em>} that is linearly separable, and suppose we do not use <em>ℓ</em><sub>1 </sub> Let |<em>SV </em>| be the number of support vectors obtained when training on the entire training set. (Recall <em>x</em><sup>(<em>i</em>) </sup>is a support vector if and only if <em>α<sub>i </sub>&gt; </em>0.) Let ˆ<em>ε</em><sub>LOOCV </sub>denote the leave one out cross validation error of our SVM. Prove that</li>

</ul>

<em>ε</em>ˆ<sub>LOOCV</sub><em>.</em>

<ul>

 <li><strong>General Case. </strong>Consider a setting similar to in part (a), except that we now run an SVM using a general (Mercer) kernel. Assume that the data is linearly separable in the high dimensional feature space corresponding to the kernel. Does the bound in part (a) on ˆ<em>ε</em><sub>LOOCV </sub>still hold? Justify your answer.</li>

</ul>

<h1>4. [12 points] MAP estimates and weight decay</h1>

Consider using a logistic regression model <em>h<sub>θ</sub></em>(<em>x</em>) = <em>g</em>(<em>θ<sup>T</sup>x</em>) where <em>g </em>is the sigmoid function, and let a training set {(<em>x</em><sup>(<em>i</em>)</sup><em>,y</em><sup>(<em>i</em>)</sup>);<em>i </em>= 1<em>,…,m</em>} be given as usual. The maximum likelihood estimate of the parameters <em>θ </em>is given by

<em>m θ </em>Y

<em>θ</em><sub>ML </sub>= argmax            <em>p</em>(<em>y</em><sup>(<em>i</em>)</sup>|<em>x</em><sup>(<em>i</em>)</sup>;<em>θ</em>)<em>.</em>

<em>i</em>=1

If we wanted to regularize logistic regression, then we might put a Bayesian prior on the parameters. Suppose we chose the prior <em>θ </em>∼ N(0<em>,τ</em><sup>2</sup><em>I</em>) (here, <em>τ &gt; </em>0, and <em>I </em>is the <em>n</em>+1-by<em>n </em>+ 1 identity matrix), and then found the MAP estimate of <em>θ </em>as:

<em>m θ    </em>Y

<em>θ</em><sub>MAP </sub>= argmax<em>p</em>(<em>θ</em>)             <em>p</em>(<em>y</em><sup>(<em>i</em>)</sup>|<em>x</em><sup>(<em>i</em>)</sup><em>,θ</em>)

<em>i</em>=1

Prove that

||<em>θ</em>MAP||2 ≤ ||<em>θ</em>ML||2

[Hint: Consider using a proof by contradiction.]

<strong>Remark. </strong>For this reason, this form of regularization is sometimes also called <strong>weight decay</strong>, since it encourages the weights (meaning parameters) to take on generally smaller values.

<h1>5.  KL divergence and Maximum Likelihood</h1>

The Kullback-Leibler (KL) divergence between two discrete-valued distributions <em>P</em>(<em>X</em>)<em>,Q</em>(<em>X</em>) is defined as follows:<sup>1</sup>

<em>K</em>

For notational convenience, we assume <em>P</em>(<em>x</em>) <em>&gt; </em>0<em>,</em>∀<em>x</em>. (Otherwise, one standard thing to do is to adopt the convention that “0log0 = 0.”) Sometimes, we also write the KL divergence as <em>KL</em>(<em>P</em>||<em>Q</em>) = <em>KL</em>(<em>P</em>(<em>X</em>)||<em>Q</em>(<em>X</em>)).

The KL divergence is an assymmetric measure of the distance between 2 probability distributions. In this problem we will prove some basic properties of KL divergence, and work out a relationship between minimizing KL divergence and the maximum likelihood estimation that we’re familiar with.

(a) Nonnegativity. Prove the following:

<h1>∀<em>P,Q KL</em>(<em>P</em>k<em>Q</em>) ≥ 0</h1>

and

<em>KL</em>(<em>P</em>k<em>Q</em>) = 0           if and only if <em>P </em>= <em>Q.</em>

[Hint: You may use the following result, called <strong>Jensen’s inequality</strong>. If <em>f </em>is a convex function, and <em>X </em>is a random variable, then <em>E</em>[<em>f</em>(<em>X</em>)] ≥ <em>f</em>(<em>E</em>[<em>X</em>]). Moreover, if <em>f </em>is strictly convex (<em>f </em>is convex if its Hessian satisfies <em>H </em>≥ 0; it is <em>strictly </em>convex if <em>H &gt; </em>0; for instance <em>f</em>(<em>x</em>) = −log<em>x </em>is strictly convex), then <em>E</em>[<em>f</em>(<em>X</em>)] = <em>f</em>(<em>E</em>[<em>X</em>]) implies that <em>X </em>= <em>E</em>[<em>X</em>] with probability 1; i.e., <em>X </em>is actually a constant.]

<ul>

 <li><strong>Chain rule for KL divergence. </strong>The KL divergence between 2 conditional distributions <em>P</em>(<em>X</em>|<em>Y </em>)<em>,Q</em>(<em>X</em>|<em>Y </em>) is defined as follows: <em>K</em>!</li>

</ul>

This can be thought of as the expected KL divergence between the corresponding conditional distributions on <em>x </em>(that is, between <em>P</em>(<em>X</em>|<em>Y </em>= <em>y</em>) and <em>Q</em>(<em>X</em>|<em>Y </em>= <em>y</em>)), where the expectation is taken over the random <em>y</em>. Prove the following chain rule for KL divergence:

<em>KL</em>(<em>P</em>(<em>X,Y </em>)k<em>Q</em>(<em>X,Y </em>)) = <em>KL</em>(<em>P</em>(<em>X</em>)k<em>Q</em>(<em>X</em>)) + <em>KL</em>(<em>P</em>(<em>Y </em>|<em>X</em>)k<em>Q</em>(<em>Y </em>|<em>X</em>))<em>.</em>

<ul>

 <li><strong>KL and maximum likelihood.</strong></li>

</ul>

Consider a density estimation problem, and suppose we are given a training set

{<em>x</em><sup>(<em>i</em>)</sup>;<em>i </em>= 1<em>,…,m</em>}. Let the empirical distribution be.

(<em>P</em>ˆ is just the uniform distribution over the training set; i.e., sampling from the empirical distribution is the same as picking a random example from the training set.) Suppose we have some family of distributions <em>P<sub>θ </sub></em>parameterized by <em>θ</em>. (If you like, think of <em>P<sub>θ</sub></em>(<em>x</em>) as an alternative notation for <em>P</em>(<em>x</em>;<em>θ</em>).) Prove that finding the maximum likelihood estimate for the parameter <em>θ </em>is equivalent to finding <em>P<sub>θ </sub></em>with minimal KL divergence from <em>P</em>ˆ. I.e. prove:

<em>m</em>

argmin<em>KL</em>(<em>P</em>ˆk<em>P<sub>θ</sub></em>) = argmax log<em>P<sub>θ</sub></em>(<em>x</em><sup>(<em>i</em>)</sup>) <em>θ                  </em><em><sub>θ </sub></em>X

<em>i</em>=1

<strong>Remark. </strong>Consider the relationship between parts (b-c) and multi-variate Bernoulli Naive Bayes parameter estimation. In the Naive Bayes model we assumed <em>P<sub>θ </sub></em>is of the following form: ). By the chain rule for KL divergence, we therefore have:

<em>n</em>

<em>KL</em>(<em><sup>P</sup></em>ˆk<em>P<sub>θ</sub></em>) = <em>KL</em>(<em><sup>P</sup></em>ˆ(<em>y</em>)k<em>p</em>(<em>y</em>)) + X<em>KL</em>(<em><sup>P</sup></em>ˆ(<em>x<sub>i</sub></em>|<em>y</em>)k<em>p</em>(<em>x<sub>i</sub></em>|<em>y</em>))<em>.</em>

<em>i</em>=1

This shows that finding the maximum likelihood/minimum KL-divergence estimate of the parameters decomposes into 2<em>n </em>+ 1 independent optimization problems: One for the class priors <em>p</em>(<em>y</em>), and one for each of the conditional distributions <em>p</em>(<em>x<sub>i</sub></em>|<em>y</em>) for each feature <em>x<sub>i </sub></em>given each of the two possible labels for <em>y</em>. Specifically, finding the maximum likelihood estimates for each of these problems individually results in also maximizing the likelihood of the joint distribution. (If you know what Bayesian networks are, a similar remark applies to parameter estimation for them.)

<h2>6.  K-means for compression</h2>

In this problem, we will apply the K-means algorithm to lossy image compression, by reducing the number of colors used in an image.

The directory /afs/ir.stanford.edu/class/cs229/ps/ps3/ contains a 512×512 image of a mandrill represented in 24-bit color. This means that, for each of the 262144 pixels in the image, there are three 8-bit numbers (each ranging from 0 to 255) that represent the red, green, and blue intensity values for that pixel. The straightforward representation of this image therefore takes about 262144 × 3 = 786432 bytes (a byte being 8 bits). To compress the image, we will use K-means to reduce the image to <em>k </em>= 16 colors. More specifically, each pixel in the image is considered a point in the three-dimensional (<em>r,g,b</em>)space. To compress the image, we will cluster these points in color-space into 16 clusters, and replace each pixel with the closest cluster centroid.

Follow the instructions below. Be warned that some of these operations can take a while (several minutes even on a fast computer)!<a href="#_ftn2" name="_ftnref2"><sup>[2]</sup></a>

<ul>

 <li>Copy mandrill-large.tiff from /afs/ir.stanford.edu/class/cs229/ps/ps3 on the leland system. Start up MATLAB, and type A = double(imread(’mandrill-large.tiff’)); to read in the image. Now, A is a “three dimensional matrix,” and A(:,:,1), A(:,:,2) and A(:,:,3) are 512×512 arrays that respectively contain the red, green, and blue values for each pixel. Enter imshow(uint8(round(A))); to display the image.</li>

 <li>Since the large image has 262144 pixels and would take a while to cluster, we will instead run vector quantization on a smaller image. Repeat (a) with mandrill-small.tiff. Treating each pixel’s (<em>r,g,b</em>) values as an element of R<sup>3</sup>, run K-means<sup>3 </sup>with 16 clusters on the pixel data from this smaller image, iterating (preferably) to convergence, but in no case for less than 30 iterations. For initialization, set each cluster centroid to the (<em>r,g,b</em>)-values of a randomly chosen pixel in the image.</li>

 <li>Take the matrix A from mandrill-large.tiff, and replace each pixel’s (<em>r,g,b</em>) values with the value of the closest cluster centroid. Display the new image, and compare it visually to the original image. Hand in all your code and a printout of your compressed image (printing on a black-and-white printer is fine).</li>

</ul>




6

<ul>

 <li>If we represent the image with these reduced (16) colors, by (approximately) whatfactor have we compressed the image?</li>

</ul>

<a href="#_ftnref1" name="_ftn1">[1]</a> If <em>P </em>and <em>Q </em>are densities for continuous-valued random variables, then the sum is replaced by an integral, and everything stated in this problem works fine as well. But for the sake of simplicity, in this problem we’ll just work with this form of KL divergence for probability mass functions/discrete-valued distributions.

<a href="#_ftnref2" name="_ftn2">[2]</a> In order to use the imread and imshow commands in octave, you have to install the Image package from octave-forge. This package and installation instructions are available at: http://octave.sourceforge.net <sup>3</sup>Please implement K-means yourself, rather than using built-in functions from, e.g., MATLAB or octave.