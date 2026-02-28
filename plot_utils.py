import matplotlib.pyplot as plt
import seaborn as sns
import os


class PlotUtils:
    def __init__(self, output_dir='output', color_theme='style1'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.color_theme = color_theme
        self._set_color_theme(color_theme)
   
        plt.style.use('seaborn-v0_8-whitegrid')
        
    def _set_color_theme(self, theme):
        theme_colors = {
            'style1': {  
                'scatter': "#1680AD",      
                'line': '#D1495B',         
                'text': '#2D3142',         
                'grid': '#E0E0E0',         
                'axes': '#333333'          
            },
            'style2': {  
                'scatter': "#2A7C22",      
                'line': '#D1495B',         
                'text': '#2D3142',         
                'grid': '#E0E0E0',
                'axes': '#333333'
            },
            'style3': {  
                'scatter': '#6A4C93',      
                'line': '#D1495B',         
                'text': '#2D3142',         
                'grid': '#E0E0E0',
                'axes': '#333333'
            },
            'style4': {  
                'scatter': "#CC3510",      
                'line': '#D1495B',         
                'text': '#2D3142',         
                'grid': '#E0E0E0',
                'axes': '#333333'
            },
            'style5': {  
                'scatter': "#FDB5B5",      
                 'line': '#D1495B',         
                'text': '#2D3142',         
                'grid': '#E0E0E0',
                'axes': '#333333'
            },
            'style6': {  
                'scatter': "#F38015F6",
                'line': '#D1495B',
                'text': '#2D3142',
                'grid': '#E0E0E0',
                'axes': '#333333'
            }
        }
        
        if theme in theme_colors:
            self.colors = theme_colors[theme]
        else:
            self.colors = theme_colors['style1']
    
    def plot_scatter(self, y_test, y_pred, mse, rmse, mae, r2, 
                     xlabel='Yield', ylabel='Predicted Yield', 
                     title_prefix='', dataset_name=''):

        fig, ax = plt.subplots(figsize=(8, 6))
        
        # use theme colors
        scatter_c = self.colors['scatter']
        line_c = self.colors['line']
        text_c = self.colors['text']
        grid_c = self.colors['grid']
        axes_c = self.colors['axes']
        
        # 调整点的大小和透明度，解决点太密集的问题
        # 5000多个点需要更小的点和更低的透明度
        scatter = ax.scatter(y_test, y_pred, alpha=0.8,  # 降低透明度
                           linewidth=0, s=40, color=scatter_c, zorder=2)  # 减小点的大小
        
        # draw y=x line with theme color
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        
        
        xlim_start = 0 
        ylim_start = 0 
        
        
        # 设置坐标轴范围，确保从最小值开始，边距为0
        ax.set_xlim([xlim_start, max_val])
        ax.set_ylim([ylim_start, max_val])
        
        # 调整y=x线的绘制范围
        ax.plot([xlim_start, max_val], 
                [xlim_start, max_val], 
                linestyle='--', linewidth=1.5, color=line_c, 
                label='y = x', zorder=1)
        
        # set labels and title with theme color
        ax.set_xlabel(xlabel, fontsize=11, color=axes_c)
        ax.set_ylabel(ylabel, fontsize=11, color=axes_c)
        
        # set title
        if dataset_name:
            title = f'{title_prefix} - {dataset_name} '
        else:
            title = f'{title_prefix} Scatter Plot'
        ax.set_title(title, fontsize=12, color=axes_c, pad=15, fontweight='medium')
        
        # set grid with theme color
        ax.grid(True, alpha=0.3, color=grid_c, linestyle='-', linewidth=0.5)
        
        # set axes spines with theme color
        ax.spines['top'].set_color(axes_c)
        ax.spines['right'].set_color(axes_c)
        ax.spines['bottom'].set_color(axes_c)
        ax.spines['left'].set_color(axes_c)
        ax.spines['top'].set_linewidth(0.5)
        ax.spines['right'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['left'].set_linewidth(0.5)
        
        # set tick params with theme color
        ax.legend(loc='lower right', frameon=True, framealpha=0.8, 
                 edgecolor='none', fontsize=10)
        
        # add metrics text 
        textstr = f'R² = {r2:.4f}\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='left',
                linespacing=1.5)
        # adjust layout
        plt.tight_layout()
        
        # save figure
        if dataset_name:
            filename = f'scatter_{dataset_name.lower().replace(" ", "_")}'
        else:
            filename = 'scatter_plot'
        self._save_plot(fig, filename)
        
        return fig, ax
    
    
    def set_theme(self, theme):
        valid_themes = ['style1', 'style2', 'style3', 'style4', 'style5']
        if theme in valid_themes:
            self.color_theme = theme
            self._set_color_theme(theme)
            print(f"配色主题已更改为: {theme}")
        else:
            print(f"警告: 配色主题 '{theme}' 不存在，使用当前主题 {self.color_theme}")
    
    def get_current_theme(self):
        return self.color_theme
    
    def get_color_palette(self):
        return {
            'style1': '蓝色主题 ',
            'style2': '绿色主题',
            'style3': '紫色主题',
            'style4': '橙色主题',
            'style5': '青色主题'
        }
    
    def _save_plot(self, fig, filename):
        # save figure
        png_path = os.path.join(self.output_dir, f'{filename}.png')
        fig.savefig(png_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"  图表已保存到: {png_path}")
        
        plt.close(fig)