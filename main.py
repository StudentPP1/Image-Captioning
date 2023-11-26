import customtkinter as ctk
from customtkinter import filedialog  
from CTkMessagebox import CTkMessagebox
import pyperclip
from model import Network
from PIL import Image


class App(ctk.CTk):
    def __init__(self):
        self.model = Network()
        super().__init__()

        ctk.set_default_color_theme("dark-blue")
        self.title("Image caption generation")
        self.geometry("700x450")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.image_find_button = ctk.CTkButton(self,
                                               corner_radius=20,
                                               height=40,
                                               text="Find the image",
                                               command=self.find_image)
        self.image_find_button.grid(row=0, column=0)
         
    def find_image(self):
        image_formats = [("Image File", '.jpg .png .jpeg')]
        image_path = filedialog.askopenfilename(filetypes=image_formats, initialdir="/", title="Please select a image")

        if image_path == "":
            CTkMessagebox(title="Error", message="Please select a image!")
            self.back()
        else:
            self.image_find_button.grid_remove()
            image = ctk.CTkImage(Image.open(image_path).resize((250, 250), Image.LANCZOS), size=(250, 250))

            self.image_label = ctk.CTkLabel(self, text="Your image:", image=image, compound="bottom",
                                            font=ctk.CTkFont(size=15), anchor=ctk.CENTER)
            self.image_label.grid(row=0, column=0, padx=10, pady=10)

            self.generate_caption(image_path)

    def generate_caption(self, image_path):
        self.progress_label = ctk.CTkLabel(self, text="", compound="bottom", font=ctk.CTkFont(size=15),
                                           anchor="n")
        self.progress_label.grid(row=1, column=0)

        self.progress_bar = ctk.CTkProgressBar(self, orientation="horizontal", determinate_speed=25,
                                               corner_radius=10)
        self.progress_bar.grid(row=2, column=0, padx=10, pady=10)
        self.progress_bar.step()

        self.progress_label.configure(text="Starting...")
        self.update()

        try:
            text = self.model.predict_caption(image_path, False, self.progress_bar, self.progress_label, self)
            self.progress_bar.stop()
            self.update()
            self.show_caption(text)
        except Exception as ex:
            print(ex)
            CTkMessagebox(title="Error", message="Try another image!")
            self.back()

    def show_caption(self, text):
        self.progress_bar.grid_remove()
        self.progress_label.grid_remove()

        self.image_label.grid(row=0, column=0, padx=10, pady=10)
        self.image_label.configure(text=f"{text}", compound="top")

        self.button_frame = ctk.CTkFrame(self)
        self.button_frame.grid(row=1, column=0)
        self.button_frame.grid_rowconfigure(0, weight=2)
        self.button_frame.grid_columnconfigure(0, weight=2)

        self.back_button = ctk.CTkButton(self.button_frame, corner_radius=20,
                                         height=40,
                                         text="Back",
                                         command=self.back)
        self.back_button.grid(row=0, column=0, padx=10, pady=10)

        self.copy_button = ctk.CTkButton(self.button_frame, corner_radius=20,
                                         height=40,
                                         text="Copy caption", command=lambda: self.copy(text))
        self.copy_button.grid(row=0, column=1, padx=10, pady=10)

    def copy(self, text):
        pyperclip.copy(text)
        CTkMessagebox(title="Info", message="Text copied!")
        
    def back(self):
        try:
            self.progress_bar.grid_remove()
            self.progress_label.grid_remove()
        except Exception as ex:
            print(ex)

        try:
            self.image_label.grid_remove()
            self.back_button.grid_remove()
            self.copy_button.grid_remove()
        except Exception as ex:
            print(ex)

        self.create_find_button(self)

    def create_find_button(self, master):
        self.image_find_button = ctk.CTkButton(master,
                                               corner_radius=20,
                                               height=40,
                                               text="Find the image",
                                               command=self.find_image)
        self.image_find_button.grid(row=0, column=0)

    def create_progress_widgets(self, master):
        self.progress_label = ctk.CTkLabel(master, text="", compound="bottom", font=ctk.CTkFont(size=15),
                                           anchor="n")
        self.progress_label.grid(row=0, column=0)
        self.progress_bar = ctk.CTkProgressBar(master, orientation="horizontal", determinate_speed=25,
                                               corner_radius=10)
        self.progress_bar.set(0)
        self.progress_bar.grid(row=1, column=0)

    def create_show_caption_widgets(self, master, text):
        self.caption_label = ctk.CTkLabel(master, text=f"{text}", compound="top",
                                          font=ctk.CTkFont(size=15), anchor=ctk.CENTER)
        self.back_button = ctk.CTkButton(master, corner_radius=20,
                                         height=40,
                                         text="Back",
                                         command=self.back)
        self.copy_button = ctk.CTkButton(master, corner_radius=20,
                                         height=40,
                                         text="Copy caption",
                                         command=lambda: self.copy(text))


if __name__ == "__main__":
    app = App()
    app.mainloop()
