from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from django.views.generic import ListView, CreateView, UpdateView, DetailView, DeleteView
from .models import Post, Comment
from django.contrib.auth.models import User
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.template.loader import render_to_string
from django.http import HttpResponse,HttpResponseRedirect,JsonResponse

# Create your views here.
def starting(request):
    return render(request,'home/starting.html',{})

class PostListView(ListView):
    model = Post
    template_name = 'home/starting.html'  # <app>/<model>_<viewtype>.html
    context_object_name = 'posts'
    ordering = ['-date_posted']
    paginate_by = 5

    def get_queryset(self):
        return Post.objects.all()

class ModelCreateView(LoginRequiredMixin, CreateView):
    model = Post
    template_name = "home/createpost.html"
    fields = ['title', 'content','images']


    def form_valid(self,form):
        form.instance.author = self.request.user
        #  self.object = Post(images=self.get_form_kwargs().get('files')['images'])
        # form.save()
        return super().form_valid(form)


    
class ModelUpdateView(LoginRequiredMixin, UserPassesTestMixin ,CreateView):
    model = Post
    template_name = "home/createpost.html"
    fields = ['title', 'content','images']
    def get_object(self):
        id_ = self.kwargs.get("id")
        return get_object_or_404(Post,id=id_)
    def form_valid(self,form):
        form.instance.author = self.request.user
        return super().form_valid(form)

    def test_func(self):
        post = self.get_object()
        if self.request.user == post.author:
            return True
        return False

# class ModelDetailView(DetailView):
#     model = Post
#     template_name = "home/detail.html"
@login_required
def postdetails(request,pk):
    posts = (Post.objects.filter(id=pk))
    user = request.user
    comments = (Comment.objects.filter(inpost=posts[0]))


    is_liked = False
    if posts[0].likes.filter(id=user.id).exists():
        is_liked= True
    total_likes = posts[0].total_likes()
    # print(total_likes)
    # print(posts)
    # print(comments)
    return render(request, 'home/detail.html', {'posts': posts, 'user': user, 'comments':comments,'is_liked':is_liked,'total_likes':total_likes})
@login_required
def commentsubmit(request,pk):
    print("hggfg")
    if request.method == 'POST' and request.is_ajax():
        posts = (Post.objects.filter(id=pk))
        user = request.user
        content = request.POST.get('content')
        comment =  Comment(author=user,inpost=posts[0],content=content)
        comment.save()
        comments = (Comment.objects.filter(inpost=posts[0]))
        context = {'posts': posts, 'user': user, 'comments':comments}
        html = render_to_string('home/comment_section.html', context, request=request)
        # print("html")
        return JsonResponse({'form': html})

        # return render(request, 'home/detail.html', {'posts': posts, 'user': user, 'comments':comments})

@login_required
def likepost(request,pk):
    posts = (Post.objects.filter(id=pk))
    user = request.user
    # print("aaya")
    is_liked = False
    if posts[0].likes.filter(id=user.id).exists():
        posts[0].likes.remove(user)
        is_liked = False
    else:
        posts[0].likes.add(user)
        is_liked = True
    total_likes = posts[0].total_likes()
    # print(total_likes)
    context ={'posts': posts[0], 'user': user,'is_liked':is_liked,'total_likes':total_likes}
    # print("bv")
    # print(request.is_ajax)
    if request.is_ajax():
        html = render_to_string('home/like_section.html', context, request=request)
        # print("html")
        return JsonResponse({'form': html})
    # print(is_liked)

    return redirect(posts[0].get_absolute_url(), is_liked=is_liked)





class ModelDeleteView(LoginRequiredMixin, UserPassesTestMixin , DeleteView):
    model = Post
    template_name = "home/delete.html"
    success_url = '/'

    def test_func(self):
        post = self.get_object()
        if self.request.user == post.author:
            return True
        return False

    
class UserListView(ListView):
    model = Post
    template_name = 'home/user_post.html'  # <app>/<model>_<viewtype>.html
    context_object_name = 'posts'
    ordering = ['-date_posted']
    paginate_by = 5
    
    def get_queryset(self):
        user = get_object_or_404(User , username=self.kwargs.get('username'))
        return Post.objects.filter(author=user).order_by('-date_posted')
    
def about(request):
    return render(request, 'home/about.html', {})

