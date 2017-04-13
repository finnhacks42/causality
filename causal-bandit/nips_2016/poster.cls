\ProvidesClass{poster}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OPTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{kvoptions}

\SetupKeyvalOptions{
	family=POSTER,
	prefix=POSTER@
}

\DeclareBoolOption{nofooter}
\DeclareStringOption{scale}
\ProcessKeyvalOptions*

\newif\iflandscape\landscapefalse
\DeclareOption{landscape}{\landscapetrue}

\ProcessOptions\relax

%\LoadClass[final,t]{beamer}
\LoadClass{beamer}

\mode<presentation>
\usepackage{beamerthemeRJH}

\iflandscape
	\usepackage[orientation=landscape,size=a0,scale=\POSTER@scale,debug]{beamerposter}
\else
	\usepackage[orientation=portrait,size=a0,scale=\POSTER@scale,debug]{beamerposter}
\fi

\usefonttheme{professionalfonts}

%\usepackage{etex}
\usepackage{tikz}
\usepackage{pgfplots}
\usepackage{pgfplotstable} 
\pgfplotsset{compat = newest} 
\usepackage{calc} 


\usetikzlibrary{trees,arrows,automata,decorations.markings,calc}

\tikzstyle{mathbox}=[baseline,inner sep=1pt,rounded corners]
\tikzstyle{every picture}+=[remember picture]


\usepackage[english]{babel}
\usepackage[latin1]{inputenc}
\usepackage{dsfont}
\usefonttheme[onlymath]{serif}

\newcommand{\defined}{\vcentcolon =}
\newcommand{\rdefined}{=\vcentcolon}

\newcommand{\eqn}[1]{\begin{align}#1\end{align}}
\newcommand{\eq}[1]{\setlength{\abovedisplayskip}{1cm}\setlength{\belowdisplayskip}{1cm}\begin{align*}#1\end{align*}}
\newcommand{\argmax}{\operatornamewithlimits{arg\,max}}
\newcommand{\argmin}{\operatornamewithlimits{arg\,min}}
\newcommand{\set}[1]{\left\{#1\right\}}
\newcommand{\ind}[1]{\mathds{1}\! \left\{#1\right\}}
\newcommand{\E}{\mathbb E}


\newcommand{\textsum}{\textstyle\sum\limits\displaystyle}

\definecolor{OliveGreen}{rgb}{0.9,0.9,1.0}
\definecolor{DullRed}{rgb}{0.7,0.1,0.1}
\definecolor{DullGreen}{rgb}{0.1,0.6,0.1}
\definecolor{DullBlue}{rgb}{0.1,0.1,0.7}
\definecolor{DullPurple}{rgb}{0.5,0.1,0.5}
\definecolor{DullYellow}{rgb}{0.3,0.3,0.0}
\definecolor{DullTeal}{rgb}{0.1,0.4,0.4}
\definecolor{LightGray}{rgb}{0.9,0.9,0.9}
\definecolor{Navy}{rgb}{0,0,0.5}

% small square bullets
%\setbeamertemplate{itemize item}{\tikz[baseline=-0.5ex]{\node[draw,fill,inner sep=0pt,minimum height=0.6ex, minimum width=0.6ex] {};}$\;$}

\setbeamercolor{item}{fg=.}
\setbeamertemplate{items}{
\setbeamercolor{math text}{fg=.}
\scalebox{0.5}{$\blacksquare$}
}
% small square bullets
\setbeamertemplate{itemize subitem}{\tikz[baseline=-0.5ex]{\node[draw,fill,inner sep=0pt,minimum height=0.6ex, minimum width=0.6ex] {};}$\;$}

\newcommand{\coloritem}[1]{
\setbeamercolor{itemize item}{fg=#1}
\setbeamercolor{enumerate item}{fg=#1}
\setbeamercolor{item}{fg=#1}
\item
\color{#1}
}


\ifPOSTER@nofooter
\setbeamertemplate{footline}{
}
\fi

