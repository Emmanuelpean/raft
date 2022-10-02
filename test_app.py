from seleniumbase import BaseCase
try:
    import pywinauto
except ModuleNotFoundError:
    pass
import cv2
import time
import core.resources


class ComponentsTest(BaseCase):
    def test_basic(self):

        # open the app and take a screenshot
        self.open("http://localhost:8501")
        time.sleep(5)

        for path in core.resources.all_files:
            self.click("section.css-1dhfpht.exg6vvm15")  # click upload button
            time.sleep(1)

            # enter
            path = path.replace('/', '\\')
            file_name = path.split('\\')[-1].split('.')[0]
            app = pywinauto.Application()
            upload_files_window = app.connect(title=u'Open', found_index=0)
            combobox = upload_files_window['OpenDialog'].ComboBox
            combobox.click()
            combobox.type_keys(path)
            combobox.type_keys(path)
            upload_files_window['OpenDialog']['Button'].click()
            time.sleep(5)  # allow raft to display the data

            # Save a screenshot of the app
            screenshot_current = "screenshots/%s_current.png" % file_name
            self.save_screenshot(screenshot_current)

            # automated visual regression testing
            # tests page has identical structure to baseline
            # https://github.com/seleniumbase/SeleniumBase/tree/master/examples/visual_testing
            # level 2 chosen, as id values dynamically generated on each page run
            self.check_window(name=file_name, level=2)

            # test screenshots look exactly the same
            original = cv2.imread("visual_baseline/test_basic/%s/baseline.png" % file_name)
            duplicate = cv2.imread(screenshot_current)

            assert original.shape == duplicate.shape
            difference = cv2.subtract(original, duplicate)
            b, g, r = cv2.split(difference)
            assert cv2.countNonZero(b) == cv2.countNonZero(g) == cv2.countNonZero(r) <= 0
