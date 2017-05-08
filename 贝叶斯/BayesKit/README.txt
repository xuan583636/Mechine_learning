This file describes a Bayes Net Toolkit
that we will refer to now as BNT.

This version is 0.1.  Let's consider this code an "alpha" version
that contains some useful functionality, but is not complete, and
is not a ready-to-use "application".

The purpose of the toolkit is to facilitate creating experimental
Bayes nets that analyze sequences of events.  The toolkit provides
code to help with the following:
(a) creating Bayes nets.  There are three classes of nodes defined,
and to construct a Bayes net, you can write code that calls the
constructors of these classes, and then you can create links among
them.
(b) displaying Bayes nets.  There is code to create new windows
and to draw Bayes nets in them.  This includes drawing the nodes,
the arcs, the labels, and various properties of nodes.
(c) propagating a-posteriori probabilities.  When one node's
probability changes, the posterior probabilities of nodes downstream
from it may need to change, too, depending on firing thresholds, etc.
There is code in the toolkit to support that.
(d) simulating events ("playing" event sequences) and having the
Bayes net respond to them.

This functionality is split over several files.  Here are the files
and the functionality that they represent.

BayesNetNode.py:  class definition for the basic node in a Bayes net.

BayesUpdating.py:  computing the a-posteriori probability of a node
 given the probabilities of its parents.

InputNode.py:  class definition for "input nodes".  InputNode is a
 subclass of BayesNetNode.  Input nodes have special features that
 allow them to recognize evidence items (using regular-expression
 pattern matching of the string descriptions of events).

OutputNode.py: class definition for "output nodes".  OutputBode
 is a subclass of BayesNetNode.  An output node can have a list of
 actions to be performed when the node's posterior probability
 exceeds a threshold

ReadWriteSigmaFiles.py:  Functionality for loading and saving
 Bayes nets in an XML format.

SampleNets.py:  Some code that constructs a sample Bayes net.
 This is called when SIGMAEditor.py is started up.

SIGMAEditor.py:  A main program that can be turned into an
 experimental application by adding menus, more code, etc.
 It has some facilities already for loading event sequence files
 and playing them.

sample-event-file.txt:  A sequence of events that exemplifies the
 format for these events.

gma-mona.igm:  A sample Bayes net in the form of an XML file.
 The SIGMAEditor program can read this type of file.

Here are some limitations of the toolkit as of 23 February 2009:

1. Users cannot yet edit Bayes nets directly in the SIGMAEditor.
Code has to be written to create new Bayes nets, at this time.
2. If you select the File menu's option to load a new Bayes net
file, you get a fixed example: gma-mona.igm.  This should be
changed in the future to bring up a file dialog box so that the
user can select the file.
3. When you "run" an event sequence in the SIGMAEditor, the
program will present each event to each input node and find out
if the input node's filter matches the evidence.  If it does match,
that fact is printed to standard output, but nothing else is done.
What should then happen is that the node's probability is updated
according to its response method, and if the new probability exceeds
the node's threshold, then its successor ("children") get their
probabilities updated, too.
4. No animation of the Bayes net is performed when an event
sequence is run.  Ideally, the diagram would be updated dynamically
to show the activity, especially when posterior probabilities of
nodes change and thresholds are exceeded.

To use the BNT, do three kinds of development:

A. create your own Bayes net whose input nodes correspond to pieces
of evidence that might be presented and that might be relevant to
drawing inferences about what's going on in the situation or process
that you are analyzing.  You do this by writing Python code that
calls constructors etc.  See the example in SampleNets.py.

B. create a sample event stream that represents a plausible sequence
of events that your system should be able to analyze.  Put this
in a file in the same format as used in sample-event-sequence.txt.

C. modify the code of BNT or add new modules as necessary to
obtain the functionality you want in your system.  This could include
code to perform actions whenever an output node's threshold is exceeded.
It could include code to generate events (rather than read them from
a file).  And it could include code to describe more clearly what is
going on whenever a node's probability is updated (e.g., what the
significance of the update is -- more certainty about something,
an indication that the weight of evidence is becoming strong, etc.)



