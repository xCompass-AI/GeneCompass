/*

    Template Name: kachari - Responsive Law Html5 Template;
    Template URI: http://hastech.company/
    Description: This is Bootstrap4 html5 template
    Author: HasTech
    Author URI: http://hastech.company/
    Version: 1.0

*/

(function ($) {
	"use Strict";
/*---------------------------------
     Mean Menu Active
-----------------------------------*/
$('.header-menu-area nav').meanmenu({
    meanMenuContainer: '.mobile-menu',
    meanScreenWidth: "991"
});
/*---------------------------------
     Sticky Menu Active
-----------------------------------*/
$(window).on('scroll',function() {
if ($(this).scrollTop() >50){  
    $('.header-sticky').addClass("is-sticky");
  }
  else{
    $('.header-sticky').removeClass("is-sticky");
  }
});
/*-------------------
  counterUp active
--------------------*/ 
$('.counter').counterUp({
    delay: 10,
    time: 1000
});
/*---------------------------------
     Team Slider Active 
----------------------------------*/
 $('.team-slider').owlCarousel({
        smartSpeed: 1000,
        items: 2,
        nav: false,
        navText: ['<i class="fa fa-angle-left"></i>', '<i class="fa fa-angle-right"></i>'],
        responsive: {
            0: {
                items: 1
            },
            992: {
                items: 2
            },
        }
    })
/*-----------------------------
    Brand Active
----------------------------------*/
 $('.brand-active').owlCarousel({
        smartSpeed: 1000,
        nav: false,
        navText: ['<i class="zmdi zmdi-chevron-left"></i>', '<i class="zmdi zmdi-chevron-right"></i>'],
        responsive: {
            0: {
                items: 2
            },
            450: {
                items: 2
            },
            600: {
                items: 3
            },
            1000: {
                items: 5
            },
            1200: {
                items: 5
            }
        }
})
/*------------------------------
   Blog Slider Active
----------------------------------*/
 $('.blog-slider-active').owlCarousel({
        smartSpeed: 1000,
        nav: true,
        loop: true,
        navText: ['<i class="fa fa-angle-left"></i>', '<i class="fa fa-angle-right"></i>'],
        responsive: {
            0: {
                items: 1
            },
            450: {
                items: 1
            },
            600: {
                items: 1
            },
            1000: {
                items: 1
            },
            1200: {
                items: 1
            }
        }
})  
/*----------------------------------
    ScrollUp Active
-----------------------------------*/
$.scrollUp({
    scrollText: '<i class="fa fa-angle-double-up"></i>',
    easingType: 'linear',
    scrollSpeed: 900,
    animation: 'fade'
});
/*----------------------------------
	 Instafeed Active 
------------------------------------*/
if($('#Instafeed').length) {
    var feed = new Instafeed({
        get: 'user',
        userId: 7093388560,
        accessToken: '7093388560.1677ed0.8e1a27120d5a4e979b1ff122d649a273',
        target: 'Instafeed',
        resolution: 'thumbnail',
        limit: 6,
        template: '<li><a href="{{link}}" target="_new"><img src="{{image}}" /></a></li>',
    });
    feed.run(); 
}
/*----------------------------------
	 Calendar Active 
------------------------------------*/
$('#my-calendar').zabuto_calendar({
    cell_border: false,
    today: true,
    show_days: true,
    weekstartson: 0,
    nav_icon: {
        prev: '<i class="fa fa-angle-left"></i>',
        next: '<i class="fa fa-angle-right"></i>'
    }
});
/* -------------------------------
	 Venobox Active
* ------------------------------*/  
$('.venobox').venobox({
    border: '10px',
    titleattr: 'data-title',
    numeratio: true,
    infinigall: true
});     
    
})(jQuery);